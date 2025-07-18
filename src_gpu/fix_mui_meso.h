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

#ifdef FIX_CLASS

FixStyle(mui/meso,FixMuiMeso)

#else

#ifndef LMP_FIX_MUI_GPU_H
#define LMP_FIX_MUI_GPU_H

#include "fix.h"
#include "mui/mui.h"
#include "pointers_meso.h"
#include <vector>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

namespace LAMMPS_NS {

using point3d = typename mui::default_config::point_type;

class FixMuiMeso : public Fix, protected MesoPointers {
public:
	FixMuiMeso(class LAMMPS *, int, char **);
	virtual ~FixMuiMeso();
	int setmask();
	virtual void init();
	virtual void post_integrate();
	virtual void end_of_step();

protected:
    mui::uniface<mui::default_config> *interface;
	int ipush_region, ifetch_region;
	double len_ratio, vel_ratio, t_ratio, step_ratio, tol;
	double push_upper, push_lower, fetch_upper, fetch_lower, sample_rc;
	bool multiscale;
	std::vector<double4> gpu_push();
	pair<vector<int>, vector<double4> > gpu_fetch_predicate();
	void gpu_fetch( pair<vector<int>, vector<double> > );
};

}

#endif
#endif
