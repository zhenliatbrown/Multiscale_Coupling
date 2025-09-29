This multiscale modeling tool will enable concurrent coupling between CPU and GPU solvers.

The demo cases are using LAMMPS as the CPU solver, and USERMESO-2.5 as the CPU solver. However, the multiscale modeling tools and corresponding algorithms developed in the demo cases are readily applied to concurrent coupling of other CPU and GPU solvers.

## Compilation Guide

### On Linux desktop

**==Important: do not install Nvidia GPU drivers with your Linux distro's software installer ==**

If you happen to have done so, we suggest you remote them. Follow Nvidia's official instruction to install CUDA Toolkit (e.g. version 9.2) including the GPU driver. Then check if the GPU(s) can be detected.

	cd ~/NVIDIA_CUDA-9.2_Samples/1_Utilities/deviceQuery
	make
	./deviceQuery

Set the required environment variales.

	export PATH=/usr/local/cuda/bin:${PATH}
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
	export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

Compile the source code as follows. To  know the compute capability version of your GPU, check https://en.wikipedia.org/wiki/CUDA

	cd <code_repository>/src
	make yes-molecule
	make yes-user-meso
	make meso ARCH=[sm_30|sm_35|sm_52|sm_60|sm_61|sm_70|...]
