/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
-------------------------------------------------------------------------*/ 
#include "stdio.h"
#include "string.h"
#include "fix_micro_multi_mui.h"
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "input.h"
#include "variable.h"
#include "error.h"
#include "force.h"
#include "region.h"
#include "stdlib.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ---------------------------------------------------------------------- */

FixMicroMultiMUI::FixMicroMultiMUI(LAMMPS *lmp, int narg, char **arg):
    Fix(lmp, narg, arg)
{
    // if (narg < 10) error->all(FLERR,"Illegal fix micro/mui command"); 

    int iarg = 3;
    domain_name = arg[iarg++];
    std::cout << domain_name << std::endl;
    
    num_interface = 2;
    for (int i = 0; i < num_interface; i++) {
        interface_names.push_back(arg[iarg++]);
        ipush_region.push_back(domain->find_region(arg[iarg++]));
        ifetch_region.push_back(domain->find_region(arg[iarg++]));
    }
    
    // std::cout << "<<<Multi>>>" << interface_names[0] << interface_names[1] << std::endl;

    // int global_size, global_rank;
    // MPI_Comm_size( lmp->world, &global_size );
    // MPI_Comm_rank( lmp->world, &global_rank );
    // std::cout << " <<<0>>>global size: " << global_size << ", rank: " << global_rank << std::endl;

    interfaces = mui::create_uniface<config>(domain_name, interface_names);//, lmp->world);
    sample_rc  = atof( arg[iarg++] );
    l_ratio=atof(arg[iarg++]);
    v_ratio=atof(arg[iarg++]);
    t_ratio=atof(arg[iarg++]);
    step_ratio=atof(arg[iarg++]);
    tol=0.0; // atof(arg[iarg++]);
              
}

/* ---------------------------------------------------------------------- */

FixMicroMultiMUI::~FixMicroMultiMUI()
{
}

/* ---------------------------------------------------------------------- */

int FixMicroMultiMUI::setmask()
{
    int mask = 0;
    mask |= POST_INTEGRATE;
    mask |= END_OF_STEP;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixMicroMultiMUI::init()
{
}

/* ---------------------------------------------------------------------- */

void FixMicroMultiMUI::post_integrate()
{
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nall = atom->nlocal;
    std::vector<double> push_val;
    std::vector<int> push_count;
    double fake[3]={0.0,0.0,0.0};
    
    int step_coupling=((update->ntimestep-1)-(update->ntimestep-1)%step_ratio)/step_ratio;
    int next_coupling=(step_coupling+1)*step_ratio;
    double bound=abs(update->ntimestep-next_coupling);
    double time = update->ntimestep * update->dt*t_ratio;
    
    for (int j = 0; j < num_interface; j++) {
        int val=0.0;
        int count=0;
        for (int i = 0; i < nall; i++) {
            if (!(mask[i] & groupbit == false)) continue;
        
            if (domain->regions[ipush_region[j]]->match(x[i][0],x[i][1],x[i][2])) {
                val += v[i][0];
                count++;
                // interface->push( "v_x", point(x[i]), real(v[i][0]) );
            }
        }
        push_val.push_back(val/count);
        push_count.push_back(count);
    }

    for (int j = 0; j < num_interface; j++) {
        if(push_count.size() == 0) continue;
        
        if (update->ntimestep == 0){
            fprintf(logfile, "<<<Multi push>>> push value %f \n", push_val[j]);}

        interfaces[j]->push("velocity_x", point(fake), real(push_val[j]));
        
        interfaces[j]->commit(time);
        interfaces[j]->barrier(time);
        interfaces[j]->forget(time);
    }

}

void FixMicroMultiMUI::end_of_step()
{
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    
    int step_coupling=update->ntimestep-update->ntimestep%step_ratio;
    double time_coupling=step_coupling*update->dt*t_ratio;
    double dt=update->dt;
    
    double sample_rc2=sample_rc/l_ratio;
    mui::sampler_shepard_quintic <config,real,real> quintic(sample_rc2);
    mui::temporal_sampler_exact <config> texact(tol);

    double time = update->ntimestep * update->dt*t_ratio;

    for (int j = 0; j < num_interface; j++) {
        double recv0 = interfaces[j]->fetch( "velocity_x", point(0.0), time_coupling, quintic, texact);
        double recv=recv0*v_ratio;

        for (int i = 0; i < nlocal; i++){
            if (!( mask[i] & groupbit )) continue;
            
            if (domain->regions[ifetch_region[j]]->match(x[i][0],x[i][1],x[i][2])  ) {
                v[i][0] += ( recv - v[i][0] ) * 1.0;
            }
        }

    }
}
