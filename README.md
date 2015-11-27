# crappy-crystals
Simulate and phase 3D diffraction volumes for crystals with translational disorder.

```
$ python crappy_crystals.py -h
usage: crappy-crystals.py [-h] config

phase a translationally disordered crystal

positional arguments:
  config      file name of the configuration file

optional arguments:
  -h, --help  show this help message and exit
```

### Requires
- python (probably >= 2.7)
- h5py 
- pyqtgraph
- scipy
- numpy


### Example
```
git clone https://github.com/andyofmelbourne/crappy-crystals.git
cd crappy-crystals
./crappy_crystals.py examples/example_duck_P1/config.ini
```

To display the ouput:
```
python utils/display.py examples/example_duck_P1/output.h5
```


### At CFEL?
Do:
```
ssh -X it-hpc-cxi01
module load python/2.7
module load opencl/intel
export PYTHONPATH=/afs/desy.de/user/a/amorgan/python_packages/lib/python2.7/site-packages/:$PYTHONPATH
./crappy_crystals.py examples/gpu_example_duck_P1/config.ini
```
Note that pyqtgraph is not installed so you will 
need to use sshfs or copy the files to your computer 
before running:
```
python utils/display.py examples/example_duck_P1/output.h5
```


### Based on the paper
- Awesome Nature paper yet to be published...
