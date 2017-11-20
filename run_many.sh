

for i in `seq 0 100`;
do 
    echo "CUDA_DEVICE=$1 python process/phase_cuda.py -c hdf5/$2/phase.ini -f hdf5/$2/$2.h5"
    CUDA_DEVICE=$1 python process/phase_cuda.py -c hdf5/$2/phase.ini -f hdf5/$2/pdb.h5
done

