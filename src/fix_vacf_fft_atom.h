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

#ifdef FIX_CLASS

FixStyle(vacf/fft/atom,FixVacfFFTAtom)

#else

#ifndef LMP_FIX_VACF_FFT_ATOM_H
#define LMP_FIX_VACF_FFT_ATOM_H

#include "stdio.h"
#include "fix.h"
#include <fstream>

namespace LAMMPS_NS {

class FixVacfFFTAtom : public Fix {
 public:
  FixVacfFFTAtom(class LAMMPS *, int, char **);
  ~FixVacfFFTAtom();
  int setmask();
  void end_of_step();

 private:
  bigint natoms_group;

  int Nstart, Nfreq, Nlength, direction, nw_max;
  double wlength;

  std::fstream fp;
  char groupname[512];

  double **data_local,**data_total;
  double ***data_init, ***vacf;
  int size, num_acf;

  void compute_fourier();
};

}

#endif
#endif
