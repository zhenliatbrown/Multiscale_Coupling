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

#include "fix_mui_meso.h"

#include "mpi.h"
#include "stdio.h"
#include "string.h"
#include "error.h"

#include "atom_vec_meso.h"
#include "engine_meso.h"
#include "atom_meso.h"
#include "comm_meso.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "force.h"
#include "input.h"
#include "region.h"
#include "update.h"
#include "variable.h"

#include "mui/mui.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

__global__ void gpu_push_gather(
	double4* __restrict push_buffer,
	uint* __restrict push_count,
	r64* __restrict coord_x,
	r64* __restrict coord_y,
	r64* __restrict coord_z,
	r64* __restrict veloc_x,
	r64* __restrict veloc_y,
	r64* __restrict veloc_z,
	int* __restrict mask,
	const r64 push_upper,
	const r64 push_lower,
	const int  groupbit,
	const int  n_atom )
{
	for(int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
		if ( ( mask[i] & groupbit ) && coord_z[i] >= push_lower && coord_z[i] <= push_upper ) {
			uint p = atomicInc( push_count, 0xFFFFFFFF );
			double4 info;
			info.x = coord_x[i];
			info.y = coord_y[i];
			info.z = coord_z[i];
			info.w = veloc_x[i];
			push_buffer[p] = info;
		}
	}
}

vector<double4> FixMuiMeso::gpu_push() {
	static int2 grid_cfg;
	static HostScalar<double4> hst_push_buffer(this->lmp,"FixMUI::push_buffer");
	static DeviceScalar<uint>  dev_push_count (this->lmp,"FixMUI::push_count");

	if ( !grid_cfg.x )
	{
		grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_push_gather, 0, cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_push_gather, cudaFuncCachePreferL1 );
		dev_push_count.grow(1);
	}
	if ( hst_push_buffer.n_elem() < atom->nlocal ) {
		hst_push_buffer.grow( atom->nlocal );
	}

	Region *region = domain->regions[ipush_region];
	push_lower = region->extent_zlo;
	push_upper = region->extent_zhi;

	dev_push_count.set( 0, meso_device->stream() );
	gpu_push_gather<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>>(
		hst_push_buffer,
		dev_push_count,
		meso_atom->dev_coord(0),
		meso_atom->dev_coord(1),
		meso_atom->dev_coord(2),
		meso_atom->dev_veloc(0),
		meso_atom->dev_veloc(1),
		meso_atom->dev_veloc(2),
		meso_atom->dev_mask,
		push_upper,
		push_lower,
		groupbit,
		atom->nlocal );

	uint n;
	dev_push_count.download( &n, 1 );
	meso_device->sync_device();
	vector<double4> result;
	for(int i=0;i<n;i++) result.push_back(hst_push_buffer[i]);
	return result;
}

__global__ void gpu_fetch_pred(
	int* __restrict pred,
	double4* __restrict loc,
	r64* __restrict coord_x,
	r64* __restrict coord_y,
	r64* __restrict coord_z,
	int* __restrict mask,
	const r64 fetch_upper,
	const r64 fetch_lower,
	const int  groupbit,
	const int  n_atom )
{
	for(int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
		if ( ( mask[i] & groupbit ) && coord_z[i] >= fetch_lower && coord_z[i] <= fetch_upper ) {
			pred[i] = 1;
			loc[i].x = coord_x[i];
			loc[i].y = coord_y[i];
			loc[i].z = coord_z[i];
		}
		else
			pred[i] = 0;
	}
}

pair<vector<int>, vector<double4> > FixMuiMeso::gpu_fetch_predicate() {
	static int2 grid_cfg;
	static HostScalar<int>     hst_fetch_pred(this->lmp,"FixMUI::fetch_pred");
	static HostScalar<double4> hst_fetch_loc(this->lmp,"FixMUI::fetch_coord");
	static vector<int> host_buffer;

	if ( !grid_cfg.x )
	{
		grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_fetch_pred, 0, cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_fetch_pred, cudaFuncCachePreferL1 );
	}
	if ( hst_fetch_pred.n_elem() < atom->nlocal ) {
		hst_fetch_pred.grow( atom->nlocal );
		hst_fetch_loc.grow( atom->nlocal );
	}


	Region *region = domain->regions[ifetch_region];
	fetch_lower = region->extent_zlo;
	fetch_upper = region->extent_zhi;

	gpu_fetch_pred<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>>(
		hst_fetch_pred,
		hst_fetch_loc,
		meso_atom->dev_coord(0),
		meso_atom->dev_coord(1),
		meso_atom->dev_coord(2),
		meso_atom->dev_mask,
		fetch_upper,
		fetch_lower,
		groupbit,
		atom->nlocal );

	meso_device->sync_device();
	vector<int> result_first;
	vector<double4> result_second;
	for(int i=0;i<hst_fetch_pred.n_elem();i++) {
		result_first.push_back( hst_fetch_pred[i] );
		result_second.push_back( hst_fetch_loc[i] );
	}
	return make_pair(result_first,result_second);
}

__global__ void gpu_scatter_fetch(
	int* __restrict pred,
	double* __restrict vres,
	r64* __restrict veloc_x,
	r64* __restrict veloc_y,
	r64* __restrict veloc_z,
	int* __restrict mask,
	const int  groupbit,
	const int  n_atom )
{
	for(int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n_atom ; i += gridDim.x * blockDim.x ) {
		if ( pred[i] ) veloc_x[i] += ( vres[i] - veloc_x[i] ) * 1.00;
	}
}

void FixMuiMeso::gpu_fetch( pair<vector<int>, vector<double> > result ) {
	static int2 grid_cfg;
	static HostScalar<int>    hst_pred(this->lmp,"FixMUI::dev_pred");
	static HostScalar<double> hst_vres(this->lmp,"FixMUI::dev_r");

	if ( !grid_cfg.x )
	{
		grid_cfg = meso_device->occu_calc.right_peak( 0, gpu_scatter_fetch, 0, cudaFuncCachePreferL1 );
		cudaFuncSetCacheConfig( gpu_scatter_fetch, cudaFuncCachePreferL1 );
	}
	if ( hst_pred.n_elem() < atom->nlocal ) {
		hst_pred.grow( atom->nlocal );
		hst_vres.grow( atom->nlocal );
	}

	for(int i=0;i<result.first.size();i++) {
		hst_pred[i] = result.first[i];
		hst_vres[i] = result.second[i];
	}
	gpu_scatter_fetch<<< grid_cfg.x, grid_cfg.y, 0, meso_device->stream() >>>(
		hst_pred,
		hst_vres,
		meso_atom->dev_veloc(0),
		meso_atom->dev_veloc(1),
		meso_atom->dev_veloc(2),
		meso_atom->dev_mask,
		groupbit,
		atom->nlocal );
}

mui::point3d point( double4 x ) {
	mui::point3d p;
	p[0] = x.x;
	p[1] = x.y;
	p[2] = x.z;
	return p;
}

FixMuiMeso::FixMuiMeso(LAMMPS *lmp, int narg, char **arg) :
	Fix(lmp, narg, arg),
	MesoPointers(lmp)
{
	if (narg != 7 && narg != 10) error->all(FLERR,"Illegal fix mui arguments");

	interface = new mui::uniface<mui::default_config>( arg[3] );
	ipush_region = domain->find_region( arg[4] );
	ifetch_region = domain->find_region( arg[5] );
	sample_rc  = atof( arg[6] );

	len_ratio = 1.0;
	vel_ratio = 1.0;
	t_ratio = 1.0;
	multiscale = false;
	if (narg == 10){
		len_ratio = atof( arg[7] );
		vel_ratio = atof( arg[8] );
		t_ratio = atof( arg[9] );
		// tol = atof( arg[10] );
		multiscale = true;
	}
	
}

FixMuiMeso::~FixMuiMeso()
{
	if ( interface ) delete interface;
}

int FixMuiMeso::setmask()
{
	int mask = 0;
	mask |= POST_INTEGRATE;
	mask |= END_OF_STEP;
	return mask;
}

void FixMuiMeso::init()
{
}

void FixMuiMeso::post_integrate()
{
	vector<double4> info = gpu_push();
	int count = info.size();
	if (count == 0) return;

	if (multiscale){
		double vel = 0.0;
		for (int i = 0; i < count; i++) {
			vel += info[i].w;
		}

		vel = vel / count;
		interface->push( "velocity_x", point3d(0.0), vel );
	}
	else{
		for (int i = 0; i < count; i++){
			interface->push( "velocity_x", point(info[i]), info[i].w);
		}
	}

	double time = update->ntimestep * update->dt * t_ratio;
	interface->commit( time );
	interface->barrier( time - 1);
	interface->forget( time - 1 );

}

void FixMuiMeso::end_of_step()
{
	int nlocal = atom->nlocal;

	mui::sampler_shepard_quintic<> quintic(sample_rc * len_ratio);
	mui::temporal_sampler_exact<> texact(tol);

	pair<vector<int>, vector<double4> > pred = gpu_fetch_predicate();
	pair<vector<int>, vector<double> > result;

	double time = update->ntimestep * update->dt * t_ratio;
	double vel = 0.0;

    // fetch average point value
	if (multiscale){
		vel = interface->fetch( "velocity_x", point3d(0.0), time, quintic, texact );
		for (int i = 0; i < nlocal; i++) {
			if ( pred.first[i] ) {
				result.second.push_back( vel/vel_ratio );
			} else
				result.second.push_back( 0.0 );
		}
	}
	else{
		for (int i = 0; i < nlocal; i++) {
			if ( pred.first[i] ) {
				vel = interface->fetch( "velocity_x", point(pred.second[i]), time, quintic, texact );
				result.second.push_back( vel/vel_ratio );
			} else
				result.second.push_back( 0.0 );
		}
	}

	result.first = pred.first;
	gpu_fetch( result );
}

