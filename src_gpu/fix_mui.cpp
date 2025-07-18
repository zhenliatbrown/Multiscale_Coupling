/* ----------------------------------------------------------------------
	 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
	 http://lammps.sandia.gov, Sandia National Laboratories
	 Steve Plimpton, sjplimp@sandia.gov

	 Copyright (2003) Sandia Corporation.	Under the terms of Contract
	 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
	 certain rights in this software.	This software is distributed under
	 the GNU General Public License.

	 See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "domain.h"
#include "fix_mui.h"
#include "force.h"
#include "input.h"
#include "region.h"
#include "respa.h"
#include "stdio.h"
#include "string.h"
#include "update.h"
#include "variable.h"

#include "mui/mui.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

FixMUI::FixMUI(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg)
{
    // if (narg != 7) error->all(FLERR,"Illegal fix mui command");
	fprintf(logfile, "print test cpu\n");
                
    interface = new mui::uniface<mui::default_config>(arg[3]);
    iregionpush = domain->find_region(arg[4]);
    iregionfetch = domain->find_region(arg[5]);
	sample_rc  = atof(arg[6]);
}

FixMUI::~FixMUI()
{
	if ( interface ) delete interface;
}

int FixMUI::setmask()
{
	int mask = 0;
	mask |= POST_INTEGRATE;
	mask |= END_OF_STEP;
	return mask;
}

void FixMUI::init()
{
}

void FixMUI::post_integrate()
{
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;


    Region *pushregion = domain->regions[iregionpush];

    for (int i = 0; i < nlocal; i++) 
    {
        if ( (mask[i] & groupbit) && pushregion->match(x[i][0], x[i][1], x[i][2]) )
        {
            interface->push( "velocity_x", point(x[i]), v[i][0] );
        }
    }

    double time = update->ntimestep * update->dt;
    interface->commit( time );
    interface->barrier( time - 1);
    interface->forget( time - 1 );
}

void FixMUI::end_of_step()
{
    double **x = atom->x;
    double **v = atom->v;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    Region *fetchregion = domain->regions[iregionfetch];

    mui::sampler_shepard_quintic <> quintic(sample_rc);
    mui::temporal_sampler_exact<>       texact(0);

    double time = update->ntimestep * update->dt;
    for (int i = 0; i < nlocal; i++) 
	{
        if (( mask[i] & groupbit ) && fetchregion->match(x[i][0], x[i][1],x[i][2]))
        {
            double res = interface->fetch( "velocity_x", point(x[i]), time, quintic, texact );
            v[i][0] += (res - v[i][0]) * 1.0;
        }
    }
}

