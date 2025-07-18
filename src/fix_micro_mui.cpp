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
#include "fix_micro_mui.h"
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

FixMicroMUI::FixMicroMUI(LAMMPS *lmp, int narg, char **arg):
    Fix(lmp, narg, arg),
    interface(NULL)
{
    if (narg < 7) error->all(FLERR,"Illegal fix micro/mui command"); 

    interface = new mui::uniface<config>( arg[3] );

    iregion1=domain->find_region(arg[4]);
    iregion2=domain->find_region(arg[5]);
    sample_rc  = atof( arg[6] );
    if (sample_rc <= 0.0) error->all(FLERR,"Illegal fix micro/mui command: sample_rc must be positive");
    multiscale = false;
    l_ratio = 1.0;
    v_ratio = 1.0;
    t_ratio = 1.0;
    step_ratio = 1;
    multiscale = false;
    if (narg == 10) {
        l_ratio=atof(arg[7]);
        v_ratio=atof(arg[8]);
        t_ratio=atof(arg[9]);
        step_ratio=atof(arg[10]);
        multiscale = true;
    }
    tol=0.0; // atof(arg[11]);
  
}

/* ---------------------------------------------------------------------- */

FixMicroMUI::~FixMicroMUI()
{
    if ( interface ) delete interface;
}

/* ---------------------------------------------------------------------- */

int FixMicroMUI::setmask()
{
    int mask = 0;
    mask |= POST_INTEGRATE;
    mask |= END_OF_STEP;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixMicroMUI::init()
{
}

/* ---------------------------------------------------------------------- */

void FixMicroMUI::post_integrate()
{
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nall = atom->nlocal;
    Region *region1=domain->regions[iregion1];
    Region *region2=domain->regions[iregion2];
    double vave = 0.0;
    int count = 0;
    double fake[3]={0.0,0.0,0.0};

    if ( multiscale ) {
        for (int i = 0; i < nall; i++){
            if ( (mask[i] & groupbit) && region1->match(x[i][0],x[i][1],x[i][2]) ){
            vave+=v[i][0];
            count++;
            }
        }
        if ( count == 0) return;
        interface->push("velocity_x",point(fake),real(vave/count));
    }
    else {
        for (int i = 0; i < nall; i++){
            if ( (mask[i] & groupbit) && region1->match(x[i][0],x[i][1],x[i][2]) ){
                interface->push( "v_x", point(x[i]), real(v[i][0]) );
            }
        }
    }

    double time = update->ntimestep * update->dt*t_ratio;
    interface->commit( time );
    interface->barrier( time-1);
    interface->forget( time - 1 );
}

void FixMicroMUI::end_of_step()
{
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    Region *region1=domain->regions[iregion1];
    Region *region2=domain->regions[iregion2];
    
    // double time = update->ntimestep * update->dt*t_ratio;
    int step_coupling = update->ntimestep - update->ntimestep%step_ratio;
    double time_coupling = step_coupling * update->dt * t_ratio;
    
    double sample_rc2=sample_rc/l_ratio;
    mui::sampler_shepard_quintic <config,real,real> quintic(sample_rc2);
    mui::temporal_sampler_exact <config> texact(tol);

    double recv = 0.0;
    if ( multiscale ){    
        double recv0 = interface->fetch( "velocity_x", point(0.0), time_coupling, quintic, texact);

        for (int i = 0; i < nlocal; i++){
            if ( ( mask[i] & groupbit ) && region2->match(x[i][0],x[i][1],x[i][2])  ) {
                v[i][0] += ( recv*v_ratio - v[i][0] ) * 1.0;
            }
        }
    }
    else {
        for (int i = 0; i < nlocal; i++){
            if ( ( mask[i] & groupbit ) && region2->match(x[i][0],x[i][1],x[i][2])  ) {
                recv = interface->fetch( "v_x", point(x[i]), time_coupling, quintic, texact);
                v[i][0] += ( recv*v_ratio - v[i][0] ) * 1.0;
            }
        }
    }

}
