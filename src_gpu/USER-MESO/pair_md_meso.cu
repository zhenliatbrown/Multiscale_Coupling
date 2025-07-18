#include "pair_md_meso.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "math.h"
#include "math_const.h"
#include "memory.h"
#include "mpi.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "stdio.h"
#include "stdlib.h"
#include "update.h"

#include "atom_meso.h"
#include "atom_vec_meso.h"
#include "comm_meso.h"
#include "neighbor_meso.h"
#include "neigh_list_meso.h"

using namespace LAMMPS_NS;
using namespace MD_COEFFICIENTS;
using namespace MathConst;

MesoPairMD::MesoPairMD( LAMMPS *lmp) : Pair( lmp ), MesoPointers( lmp ),
    dev_coefficients( lmp, "MesoPairMD::dev_coefficients" )
{
    split_flag = 0;
    coeff_ready = false;
}

MesoPairMD::~MesoPairMD()
{
    if ( !allocated ) return;

    memory->destroy(setflag);
    
    memory->destroy(cut);
    memory->destroy(cutsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
}

void MesoPairMD::allocate()
{
    allocated = 1;
    int n = atom->ntypes;

    memory->create( setflag, n + 1, n + 1, "pair::setflag");

    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            setflag[i][j];

    memory->create( cut, n + 1, n + 1, "pair::cut");
    memory->create( cutsq, n + 1, n + 1, "pair::cutsq");

    memory->create( epsilon, n + 1, n + 1, "pair::epsilon");
    memory->create( sigma, n + 1, n + 1, "pair::sigma");
    memory->create( lj1, n + 1, n + 1, "pair::lj1");
    memory->create( lj2, n + 1, n + 1, "pair::lj2");
    memory->create( lj3, n + 1, n + 1, "pair::lj3");
    memory->create( lj4, n + 1, n + 1, "pair::lj4");
    memory->create( offset, n + 1, n + 1, "pair::offset");

    dev_coefficients.grow( n * n * n_coeff );
}

void MesoPairMD::prepare_coeff()
{
    if (coeff_ready) return;
    if (!allocated) allocate();

    int n = atom->ntypes;
    coeff_table.resize( n * n * n_coeff);

    for ( int i = 1; i <= n; i++ )
    {
        for ( int j = 1; j <= n; j++)
        {
            int cid = (i - 1) * n + (j - 1);
            coeff_table[ cid * n_coeff + p_cut ] = cut[i][j];
            coeff_table[ cid * n_coeff + p_cutsq] = cutsq[i][j];
            coeff_table[ cid * n_coeff + p_epsilon] = epsilon[i][j];
            coeff_table[ cid * n_coeff + p_sigma] = sigma[i][j];
            coeff_table[ cid * n_coeff + p_lj1] = lj1[i][j];
            coeff_table[ cid * n_coeff + p_lj2] = lj2[i][j];
            coeff_table[ cid * n_coeff + p_lj3] = lj3[i][j];
            coeff_table[ cid * n_coeff + p_lj4] = lj4[i][j];
            coeff_table[ cid * n_coeff + p_offset] = offset[i][j];
        }
    }

    dev_coefficients.upload( &coeff_table[0], coeff_table.size(), meso_device->stream() );
    coeff_ready = true;
}

template<int evflag>
__global__ void gpu_pair_md(
    texobj tex_coord, texobj tex_veloc,
    r64* __restrict force_x,   r64* __restrict force_y,   r64* __restrict force_z,
    r64* __restrict virial_xx, r64* __restrict virial_yy, r64* __restrict virial_zz,
    r64* __restrict virial_xy, r64* __restrict virial_xz, r64* __restrict virial_yz,
    int* __restrict pair_count, int* __restrict pair_table,
    r64* __restrict e_pair, r64* __restrict coefficients,
    const int pair_padding,
    const int n_type,
    const int p_beg,
    const int p_end,
    const int n_part)
{
    int block_per_part = gridDim.x / n_part;
    int part_id = blockIdx.x / block_per_part;
    if ( part_id >= n_part ) return;
    int part_size = block_per_part * blockDim.x;
    int id_in_partition = blockIdx.x % block_per_part * blockDim.x + threadIdx.x;
    
    extern __shared__ r64 coeffs[];
    for ( int p = threadIdx.x; p < n_type * n_type * n_coeff; p += blockDim.x )
        coeffs[p] = coefficients[p];
    __syncthreads();

    for ( int iter = id_in_partition; ; iter += part_size)
    {
        int i = (p_beg & WARPALIGN) + iter;
        if (i >= p_end) break;
        if (i >= p_beg)
        {
            f3u coord1 = tex1Dfetch<float4>( tex_coord, i);
            f3u veloc1 = tex1Dfetch<float4>( tex_veloc, i);
            int n_pair = pair_count[i];
            int *p_pair = pair_table + ( i - __laneid() + part_id ) * pair_padding + __laneid();
            r64 fx = 0., fy = 0., fz = 0.;
            r64 vrxx = 0., vryy = 0., vrzz = 0.;
            r64 vrxy = 0., vrxz = 0., vryz = 0.;
            r64 energy = 0.;

            for ( int p = part_id; p < n_pair; p += n_part)
            {
                int j = __lds( p_pair );
                p_pair += pair_padding * n_part;
                if ( (p & 31) + n_part >= WARPSZ ) p_pair -= WARPSZ * pair_padding - WARPSZ;

                f3u coord2 = tex1Dfetch<float4>( tex_coord, j);
                r64 dx = coord1.x - coord2.x;
                r64 dy = coord1.y - coord2.y;
                r64 dz = coord1.z - coord2.z;
                r64 rsq = dx * dx + dy * dy + dz * dz;
                r64 *coeff_ij = coeffs + ( coord1.i * n_type + coord2.i ) * n_coeff;

                if ( rsq < coeff_ij[p_cutsq] && rsq >= EPSILON_SQ)
                {
                    f3u veloc2 = tex1Dfetch<float4>( tex_veloc, j);
                    r64 r2inv = 1.0/rsq;
                    r64 r6inv = r2inv * r2inv * r2inv;
                    r64 forcelj = r6inv * (coeff_ij[p_lj1] * r6inv - coeff_ij[p_lj2]);

                    r64 fpair = forcelj * r2inv;
                    fx += fpair * dx;
                    fy += fpair * dy;
                    fz += fpair * dz;

                    if ( evflag )
                    {
                        vrxx += dx * dx * fpair;
                        vryy += dy * dy * fpair;
                        vrzz += dz * dz * fpair;
                        vrxy += dx * dy * fpair;
                        vrxz += dx * dz * fpair;
                        vryz += dy * dz * fpair;
                        energy = r6inv * (coeff_ij[p_lj3] * r6inv - coeff_ij[p_lj4]) - coeff_ij[p_offset];
                    }
                }
            }

            if ( n_part == 1)
            {
                force_x[i] += fx;
                force_y[i] += fy;
                force_z[i] += fz;
                
                if (evflag)
                {
                    virial_xx[i] += vrxx * 0.5;
                    virial_yy[i] += vryy * 0.5;
                    virial_zz[i] += vrzz * 0.5;
                    virial_xy[i] += vrxy * 0.5;
                    virial_xz[i] += vrxz * 0.5;
                    virial_yz[i] += vryz * 0.5;
                    e_pair[i] = energy * 0.5;
                }
            }
            else
            {
                atomic_add( force_x + i, fx);
                atomic_add( force_y + i, fy);
                atomic_add( force_z + i, fz);
                
                if (evflag)
                {
                    atomic_add( virial_xx + i, vrxx * 0.5 );
                    atomic_add( virial_yy + i, vryy * 0.5 );
                    atomic_add( virial_zz + i, vrzz * 0.5 );
                    atomic_add( virial_xy + i, vrxy * 0.5 );
                    atomic_add( virial_xz + i, vrxz * 0.5 );
                    atomic_add( virial_yz + i, vryz * 0.5 );
                    atomic_add( e_pair + i, energy * 0.5);
                }
            }
        }
    }
}

void MesoPairMD::compute_kernel( int eflag, int vflag, int p_beg, int p_end )
{
    if ( !coeff_ready ) prepare_coeff();
    MesoNeighList *dlist = meso_neighbor->lists_device[ list-> index ];

    int shared_mem_size = atom->ntypes * atom->ntypes * n_coeff * sizeof( r64 );

    if ( eflag || vflag )
    {
        // evaluate force, energy and virial
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_pair_md<1>, shared_mem_size );
        gpu_pair_md<1> <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>>(
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged,
            meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
            meso_atom->dev_virial(0), meso_atom->dev_virial(1), meso_atom->dev_virial(2),
            meso_atom->dev_virial(3), meso_atom->dev_virial(4), meso_atom->dev_virial(5),
            dlist->dev_pair_count_core, dlist->dev_pair_table, meso_atom->dev_e_pair,
            dev_coefficients, dlist->n_col, atom->ntypes,
            p_beg, p_end, grid_cfg.partition( p_end - p_beg, WARPSZ)
        );
    }
    else
    {
        // evaluate force only
        static GridConfig grid_cfg = meso_device->configure_kernel( gpu_pair_md<0>, shared_mem_size );
        gpu_pair_md<0> <<< grid_cfg.x, grid_cfg.y, shared_mem_size, meso_device->stream() >>>(
            meso_atom->tex_coord_merged, meso_atom->tex_veloc_merged,
            meso_atom->dev_force(0), meso_atom->dev_force(1), meso_atom->dev_force(2),
            meso_atom->dev_virial(0), meso_atom->dev_virial(1), meso_atom->dev_virial(2),
            meso_atom->dev_virial(3), meso_atom->dev_virial(4), meso_atom->dev_virial(5),
            dlist->dev_pair_count_core, dlist->dev_pair_table, meso_atom->dev_e_pair,
            dev_coefficients, dlist->n_col, atom->ntypes,
            p_beg, p_end, grid_cfg.partition( p_end - p_beg, WARPSZ)
        );
    }
}

void MesoPairMD::compute_bulk( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BULK, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairMD::compute_border( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::BORDER, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::GHOST, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}

void MesoPairMD::compute( int eflag, int vflag )
{
    int p_beg, p_end, c_beg, c_end;
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::LOCAL, p_beg, p_end );
    meso_atom->meso_avec->resolve_work_range( AtomAttribute::ALL, c_beg, c_end );
    meso_atom->meso_avec->dp2sp_merged( seed_now(), c_beg, c_end, true ); // convert coordinates data to r32
    compute_kernel( eflag, vflag, p_beg, p_end );
}


uint MesoPairMD::seed_now()
{
    return premix_TEA<64>( seed, update->ntimestep );
}

void MesoPairMD::settings( int narg, char **arg )
{
    if ( narg < 1) error->all( FLERR, "Illegal pair_style command" );

    cut_global = atof( arg[0] );
    if ( narg > 1) seed = atoi( arg[1] );

    // reset cutoffs that have been explicitly set
    if ( allocated )
    {
        for ( int i = 1; i <= atom->ntypes; i++ )
            for ( int j = 1; j <= atom->ntypes; j++ )
                if ( setflag[i][j] )
                    cut[i][j] = cut_global;
    }
}

void MesoPairMD::coeff( int narg, char **arg )
{
    if ( narg <4 || narg > 5)
        error->all( FLERR, "Incorrect args for pair coefficients" );
    if ( !allocated ) allocate();

    int ilo, ihi, jlo, jhi;
    force->bounds( arg[0], atom->ntypes, ilo, ihi );
    force->bounds( arg[1], atom->ntypes, jlo, jhi );

    double epsilon_one = atof( arg[2] );
    double sigma_one = atof( arg[3] );
    double cut_one = cut_global;
    if ( narg == 5 ) cut_one = atof( arg[4] );

    int count = 0;
    for ( int i = ilo; i <= ihi; i++)
    {
        for ( int j = MAX( jlo, i ); j <= jhi; j++)
        {
            epsilon[i][j] = epsilon_one;
            sigma[i][j] = sigma_one;
            cut[i][j] = cut_one;
            cutsq[i][j] = cut_one * cut_one;
            setflag[i][j] = 1;
            count++;
        }
    }

    if ( count == 0 ) error->all( FLERR, "Incorrect args for pair coefficients" );
}

/* ----------------------------------------------------------------------
 init specific to MD pair style
 ------------------------------------------------------------------------- */

void MesoPairMD::init_style()
{
    int i = neighbor->request( this );
    neighbor->requests[i]->cudable = 1;
    neighbor->requests[i]->newton = 2;
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double MesoPairMD::init_one( int i, int j )
{
    if ( setflag[i][j] == 0 )
    {
        epsilon[i][j] = mix_energy( epsilon[i][i],epsilon[j][j],epsilon[i][i],sigma[j][j] );
        sigma[i][j] = mix_distance( sigma[i][i], sigma[j][j] );
        cut[i][j] = mix_distance( cut[i][i], cut[j][j] );
    }

    lj1[i][j] = 48.0 * epsilon[i][j] * pow( sigma[i][j], 12.0 );
    lj2[i][j] = 24.0 * epsilon[i][j] * pow( sigma[i][j], 6.0 );
    lj3[i][j] = 4.0 * epsilon[i][j] * pow( sigma[i][j], 12.0 );
    lj4[i][j] = 4.0 * epsilon[i][j] * pow( sigma[i][j], 6.0 );
    
    if ( offset_flag ){
        double ratio = sigma[i][j] / cut[i][j];
        offset[i][j] = 4.0 * epsilon[i][j] * ( pow(ratio, 12.0) - pow(ratio, 6.0));
    } else offset[i][j] = 0.0;

    lj1[j][i] = lj1[i][j];
    lj2[j][i] = lj2[i][j];
    lj3[j][i] = lj3[i][j];
    lj4[j][i] = lj4[i][j];
    offset[j][i] = offset[i][j];

    if (tail_flag) {
        int *type = atom->type;
        int nlocal = atom->nlocal;

        double count[2],all[2];
        count[0] = count[1] = 0.0;

        for (int k = 0; k < nlocal; k++) {
            if (type[k] == i) count[0] += 1.0;
            if (type[k] == j) count[1] += 1.0;
        }
        
        MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

        double sig2 = sigma[i][j]*sigma[i][j];
        double sig6 = sig2*sig2*sig2;
        double rc3 = cut[i][j]*cut[i][j]*cut[i][j];
        double rc6 = rc3*rc3;
        double rc9 = rc3*rc6;
        etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] * sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
        ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] * sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
    }

    return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void MesoPairMD::write_restart(FILE *fp)
{
    write_restart_settings(fp);

    for ( int i = 1; i <= atom->ntypes; i++)
    {
        for ( int j = i; j <= atom->ntypes; j++) 
        {
            fwrite( &setflag[i][j], sizeof(int), 1, fp );
            if (setflag[i][j]) 
            {
                fwrite( &epsilon[i][j], sizeof(double), 1, fp);
                fwrite( &sigma[i][j], sizeof(double), 1, fp);
                fwrite( &cut[i][j], sizeof(double), 1, fp);
            }
        }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void MesoPairMD::read_restart(FILE *fp)
{
    read_restart_settings( fp );
    allocate();

    int me = comm->me;
    for ( int i = 1; i <= atom->ntypes; i++ )
    {
        for ( int j = i; j <= atom->ntypes; j++ )
        {
            if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
            MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);

            if (setflag[i][j]) 
            {
                if (me == 0) 
                {
                    fread( &epsilon[i][j], sizeof(double), 1, fp);
                    fread( &sigma[i][j], sizeof(double), 1, fp);
                    fread( &cut[i][j], sizeof(double), 1, fp);
                }
                MPI_Bcast( &epsilon[i][j], 1, MPI_DOUBLE, 0, world);
                MPI_Bcast( &sigma[i][j], 1, MPI_DOUBLE, 0, world);
                MPI_Bcast( &cut[i][j], 1, MPI_DOUBLE, 0, world);
            }
        }
    }
}

/* ----------------------------------------------------------------------
 proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void MesoPairMD::write_restart_settings( FILE *fp )
{
    fwrite( &cut_global, sizeof( double ), 1, fp );
    fwrite( &offset_flag, sizeof( int ), 1, fp);
    fwrite( &seed, sizeof( int ), 1, fp );
    fwrite( &mix_flag, sizeof( int ), 1, fp );
}

/* ----------------------------------------------------------------------
 proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void MesoPairMD::read_restart_settings( FILE *fp )
{
    if( comm->me == 0 ) {
        fread( &cut_global, sizeof( double ), 1, fp );
        fread( &offset_flag, sizeof( int ), 1, fp);
        fread( &seed, sizeof( int ), 1, fp );
        fread( &mix_flag, sizeof( int ), 1, fp );
    }
    MPI_Bcast( &cut_global, 1, MPI_DOUBLE, 0, world );
    MPI_Bcast( &offset_flag, 1, MPI_INT, 0, world );
    MPI_Bcast( &seed, 1, MPI_INT, 0, world );
    MPI_Bcast( &mix_flag, 1, MPI_INT, 0, world );
}

double MesoPairMD::single( int i, int j, int itype, int jtype, double rsq,
                           double factor_coul, double factor_lj, double &fforce)
{
    double r2inv, r6inv, forcelj, philj;

    r2inv = 1.0 / rsq;
    r6inv = r2inv * r2inv * r2inv;
    forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
    fforce = factor_lj * forcelj * r2inv; 

    philj = r6inv * (lj3[itype][jtype]*r6inv-lj4[itype][jtype]) - offset[itype][jtype];
    return factor_lj * philj;
}
