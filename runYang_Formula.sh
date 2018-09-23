export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


n=10000
aca=2
ACC=1e-3
LEAF=128
nmpi=2
export EXEC=./build/examples/DenseToeplitzQChem_HODLR
mpirun -n 2 ${EXEC} ${n} ${aca} \
--hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a toeplitzQchem.out



# srun -n 2 -c 2 --cpu_bind=cores ./build/examples/DenseToeplitzQChem 10000 --hss_leaf_size 128 --hss_rel_tol 1e-3 --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a toeplitzQchem.out


h=0.1
lambda=10
ninc=1
aca=2
ACC=1e-2
LEAF=128
nmpi=2
export EXEC=./build/examples/KernelRegressionMPI
export DATA=/home/administrator/Desktop/research/ML/SUSY/susy_10Kn
# srun -n ${nmpi} -c 2 --cpu_bind=cores ${EXEC} ${DATA} 8 ${h} 1 cob ${lambda} ${nmpi} ${ninc} ${aca} test \
#--hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a susy_kernel10K.out
mpirun -n 2 ${EXEC} ${DATA} 8 ${h} 1 cob ${lambda} ${nmpi} ${ninc} ${aca} test \
--hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a susy_kernel10K.out
