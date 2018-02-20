#!/bin/sh

# loop over number
#   make a slurm file that:
#       runs the even recon
#       runs the odd recon
#       makes the figures

# make the SLURMFILE
JOBNAME=crappy_crystal_phasing_$1
SLURMFILE=hdf5/${1}/SLURMFILE.sh

echo "#!/bin/sh" > $SLURMFILE
echo "#SBATCH --workdir    /gpfs/cfel/cxi/scratch/user/amorgan/2016/crappy_crystals/crappy_crystals/" >> $SLURMFILE
echo "#SBATCH --time       2:00:00" >> $SLURMFILE
echo "#SBATCH --partition  cfel" >> $SLURMFILE
echo "#SBATCH --nice=0" >> $SLURMFILE
echo "#SBATCH --nodes      1" >> $SLURMFILE
echo "#SBATCH --job-name   ${JOBNAME}" >> $SLURMFILE
echo "#SBATCH --mail-type  END" >> $SLURMFILE
echo "#SBATCH --mail-user  andrew.morgan@desy.de" >> $SLURMFILE
echo "#SBATCH --output     hdf5/${1}/${JOBNAME}.sh.out" >> $SLURMFILE
echo "#SBATCH --error      hdf5/${1}/${JOBNAME}.sh.err" >> $SLURMFILE
echo "#SBATCH --mincpus=64" >> $SLURMFILE


echo "source /etc/profile.d/modules.sh" >> $SLURMFILE
echo "module load anaconda/2" >> $SLURMFILE
echo "source activate phasing" >> $SLURMFILE

CMD="mpirun -np 64 python process/phase_mpi.py -f hdf5/${1}/${1}.h5 -c hdf5/${1}/phase.ini"
echo $CMD
echo $CMD >> $SLURMFILE

# submit the SLURMFILE
echo "sbatch ${SLURMFILE}"
sbatch ${SLURMFILE}
