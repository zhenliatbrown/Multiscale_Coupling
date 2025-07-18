cd ../src_gpu

# make clean-all
# make yes-molecule
# make yes-user-meso
make meso ARCH=sm_89 -j16

cd ../dpd_coupling


cd ../src
make mpi -j16
cd ../dpd_coupling