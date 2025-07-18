/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Zhen Li at Brown University
   Email: zhen_li@brown.edu

   This fix is designed for computing the velocity correlation functions 
   (sound mode and shear mode) in the Fourier space.

   Remark: This routine considers multiple wave numbers, starting with 
           1, 2, 3, ... , until nw_max.

   Ref: X. Bian, Z. Li, M. Deng and G.E. Karniadakis. 
        Fluctuating hydrodynamics in periodic domains and heterogeneous 
        adjacent multidomains: Thermal equilibrium. 
        Physical Review E, 92(5): 053302, 2015.

   e.g.
   fix  1  all vacf/fft/atom 0 10 3000 0 2 10.0
------------------------------------------------------------------------- */

#include "atom.h"
#include "stdlib.h"
#include "string.h"
#include "unistd.h"
#include "fix_vacf_fft_atom.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "group.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "domain.h"
#include "comm.h"
#include "math.h"
#include <mpi.h>
#include <fstream>

#define const_pi 3.141592653589793
#define const_pi_sqrt 1.772453850905516

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ---------------------------------------------------------------------- */
FixVacfFFTAtom::FixVacfFFTAtom(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 9) error->all(FLERR,"Illegal fix vacf/fft/atom command");

  sprintf(groupname,"%s", arg[1]);
  Nstart = force->inumeric(FLERR,arg[3]);
  Nfreq = force->inumeric(FLERR,arg[4]);
  Nlength = force->inumeric(FLERR,arg[5]);
  direction  = force->inumeric(FLERR,arg[6]);
  nw_max  = force->inumeric(FLERR,arg[7]);
  wlength = force->numeric(FLERR,arg[8]);

  if ( Nfreq > Nlength ) {
    error->all(FLERR, "Illegal for Nfreq > Nlength");
  }
  if ( nw_max < 1 ) {
    error->all(FLERR, "Illegal for nw_max < 1");
  }

  natoms_group = group->count(igroup);

  //*********************************************************
  // determine the size of correlation function: 
  // v parallel - v parallel (sound mode)
  // v perpendicular - v perpendicular (shear mode)
  //*********************************************************
  num_acf = 2;

  memory->create(data_local,nw_max,3,"fix_vacf_fft_atom:data_local");
  memory->create(data_total,nw_max,3,"fix_vacf_fft_atom:data_total");

  for (int i = 0; i < nw_max; i++)
  for (int j = 0; j < 3; j++) {
    data_local[i][j] = 0.0;
    data_total[i][j] = 0.0;
  }

  compute_fourier();

  size = int(Nlength/Nfreq);
  Nlength = Nfreq * size; 

  memory->create(data_init,nw_max,size,3,"fix_vacf_fft_atom:data_init");
  memory->create(vacf,nw_max,size,1+num_acf,"fix_vacf_fft_atom:vacf");
  for(int i = 0; i < nw_max; i++)
  for(int j = 0; j < size; j++)
  for(int k = 0; k <= num_acf; k++)    
    vacf[i][j][k] = 0.0;

//  save initial information at time step [][0][]
  for (int i = 0; i < nw_max; i++)
  for (int j = 0; j < 3; j++) 
    data_init[i][0][j] = data_total[i][j];
}

/* ---------------------------------------------------------------------- */

FixVacfFFTAtom::~FixVacfFFTAtom()
{
  memory->destroy(data_local);
  memory->destroy(data_total);
  memory->destroy(data_init);
  memory->destroy(vacf);
}

/* ---------------------------------------------------------------------- */

int FixVacfFFTAtom::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}


/* ---------------------------------------------------------------------- */

void FixVacfFFTAtom::end_of_step()
{
  bigint ntimestep = update->ntimestep;
  int step, loop;

  if( ntimestep >= Nstart ) {
    step = (ntimestep - Nstart) % Nlength;
    loop = (ntimestep - Nstart) / Nlength;
    if( step % Nfreq == 0 ) {
      double vpara_sq,vperp_sq;
  
      compute_fourier();

      if(comm->me == 0) {
        int index=step/Nfreq;
        for (int i = 0; i < nw_max; i++)
        for (int j = 0; j < 3; j++)
          data_init[i][index][j] = data_total[i][j];
               
        for(int i = 0; i < nw_max; i++){
          for(int k=0; k <= index; k++) {
            switch ( direction ) {
            //wave in x direction
            case 0:
                vpara_sq  = data_init[i][k][0] * data_total[i][0];
                vperp_sq  = 0.5*(data_init[i][k][1]*data_total[i][1] + data_init[i][k][2]*data_total[i][2]);
            break;
            //wave in y direction
            case 1:
                vpara_sq  = data_init[i][k][1] * data_total[i][1];
                vperp_sq  = 0.5*(data_init[i][k][0]*data_total[i][0] + data_init[i][k][2]*data_total[i][2]);
            break;
            //wave in z direction
            case 2:
                vpara_sq  = data_init[i][k][2] * data_total[i][2];
                vperp_sq  = 0.5*(data_init[i][k][0]*data_total[i][0] + data_init[i][k][1]*data_total[i][1]);
            break;
            default:
                printf("No such direction!\n");
            }

	        vacf[i][index-k][0] += vpara_sq;
            vacf[i][index-k][1] += vperp_sq;
	        vacf[i][index-k][2] += 1;
	      }

          if( loop > 0 )
          for(int k = index+1; k < size; k++) {
            switch ( direction ) {
            //wave in x direction
            case 0:
                vpara_sq  = data_init[i][k][0] * data_total[i][0];
                vperp_sq  = 0.5*(data_init[i][k][1]*data_total[i][1] + data_init[i][k][2]*data_total[i][2]);
            break;
            //wave in y direction
            case 1:
                vpara_sq  = data_init[i][k][1] * data_total[i][1];
                vperp_sq  = 0.5*(data_init[i][k][0]*data_total[i][0] + data_init[i][k][2]*data_total[i][2]);
            break;
            //wave in z direction
            case 2:
                vpara_sq  = data_init[i][k][2] * data_total[i][2];
                vperp_sq  = 0.5*(data_init[i][k][0]*data_total[i][0] + data_init[i][k][1]*data_total[i][1]);
            break;
            default:
                printf("No such direction!\n");
            }

	        vacf[i][size+index-k][0] += vpara_sq;
            vacf[i][size+index-k][1] += vperp_sq;
	        vacf[i][size+index-k][2] += 1;          
          }
        }
      }
    }
 
    if(ntimestep == update->laststep && comm->me == 0) {
      char buff[512];
      for (int i = 0; i < nw_max; i++ ) {
        switch ( direction ) {
        case 0:
          sprintf(buff,"%s%s%s%i%s", "acf_",groupname,"_x_k",i+1,".dat");
        break;
        case 1:
          sprintf(buff,"%s%s%s%i%s", "acf_",groupname,"_y_k",i+1,".dat");
        break;
        case 2:
          sprintf(buff,"%s%s%s%i%s", "acf_",groupname,"_z_k",i+1,".dat");
        break;
        default:
          printf("No such direction!\n");
        }
      
        int n = strlen(buff)+1;
        char filename[n];
        strcpy(filename,buff);
        fp.open(filename,ios::out);

        fp << "#variables: step, time, vpara-vpara, vperp-vperp, counter\n";

        for(int j = 0; j < size; j++) { 
          int output_step = j * Nfreq;
          double output_time = output_step * update->dt;

          if(vacf[i][j][2] > 0)
          fp << output_step << "  " << output_time << "  " << vacf[i][j][0]/vacf[i][j][2]*vacf[i][0][2]/vacf[i][0][0] << " " << vacf[i][j][1]/vacf[i][j][2]*vacf[i][0][2]/vacf[i][0][1] << " " << vacf[i][j][2] << endl;
        }
        fp.close();
      }
    }
  }      
}


void FixVacfFFTAtom::compute_fourier()
{
  for (int i = 0; i < nw_max; i++)
  for (int j = 0; j < 3; j++)
    data_local[i][j] = 0.0;

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
  if (mask[i] & groupbit)
  for (int j = 0; j < nw_max; j++) {
    double s = sin(2.0*const_pi*(j+1)*x[i][direction]/wlength);
    data_local[j][0] += v[i][0]*s;
    data_local[j][1] += v[i][1]*s;
    data_local[j][2] += v[i][2]*s;
  }

  MPI_Reduce(*data_local,*data_total,nw_max*3,MPI_DOUBLE,MPI_SUM,0,world);

  if(comm->me == 0)
  for (int i = 0; i < nw_max; i++) 
  for (int j = 0; j < 3; j++)
    data_total[i][j] /= natoms_group;
}
