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

### Installation as local user on Linux
```
$ git clone https://github.com/andyofmelbourne/crappy_crystals.git 
```

### Requires
- python (probably = 2.7)
- h5py 
- pyqtgraph
- scipy
- numpy
- https://github.com/andyofmelbourne/3D-Phasing.git


### Example
```
$ cp -r crappy_crystals/examples .
$ crappy_crystals/run_crystals.py examples/duck/config.ini
```

To display the output:
```
$ crappy_crystals.py -d examples/example_duck_P1/config.ini
```

#### Getting afnumpy working on the CFEL maxwell machines
```
ssh -X max-cfelg
module load python/2.7
module load opencl/intel
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/amorgan/.local/lib/
```

