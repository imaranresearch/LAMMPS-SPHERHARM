#! /bin/sh
cd src/
#make no-all
#make clean-all
make yes-RIGID
make yes-GRANULAR
make yes-SHDEM
#make yes-SPHERHARMTEST
make yes-MOLECULE
make yes-ASPHERE
#make yes-VTK
cd STUBS/
make
cd ..
make -j 8  mpi
