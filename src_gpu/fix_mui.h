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

FixStyle(mui,FixMUI)

#else

#ifndef LMP_FIX_MUI_H
#define LMP_FIX_MUI_H

#include "fix.h"
#include "mui/mui.h"
#include <utility>

namespace LAMMPS_NS {

using namespace mui;
using point = typename default_config::point_type;

class FixMUI : public Fix
{
	public:
		FixMUI(class LAMMPS *, int, char **);
		virtual ~FixMUI();
		int setmask();
		virtual void init();
		virtual void post_integrate();
		virtual void end_of_step();

	protected:
		mui::uniface<mui::default_config> *interface;
        int iregionpush;
        int iregionfetch;
        double sample_rc;

		// double send_upper, send_lower, recv_upper, recv_lower;
};

}

#endif
#endif
