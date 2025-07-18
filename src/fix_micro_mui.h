/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(micro/mui,FixMicroMUI)

#else

#ifndef LMP_FIX_MICRO_MUI_H
#define LMP_FIX_MICRO_MUI_H

#include "fix.h"
#include "mui/mui.h"

namespace LAMMPS_NS {

using namespace mui;
using config = default_config;
using real   = typename config::REAL;
using point  = typename config::point_type;

class FixMicroMUI : public Fix {
 public:
  FixMicroMUI(class LAMMPS *, int, char **);
  virtual ~FixMicroMUI();
  int setmask();
  void init();
  void post_integrate();
  void end_of_step();

 protected:
  mui::uniface<config> *interface;
  real sample_rc;
  int iregion1,iregion2,step_ratio;
  real l_ratio,v_ratio,t_ratio,tol;
  bool multiscale = false;
};

}

#endif
#endif
