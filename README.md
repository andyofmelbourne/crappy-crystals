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
$ git clone https://github.com/andyofmelbourne/crappy-crystals.git ~/.local/lib/python2.7/site-packages/crappy_crystals
$ export PATH=$PATH:~/.local/lib/python2.7/site-packages/crappy_crystals
```

or (more perminantly) add

```
export PATH=$HOME/.local/lib/python2.7/site-packages/crappy_crystals:$PATH
```
to your ~/.bashrc or ~/.zshrc file.

### Requires
- python (probably >= 2.7)
- h5py 
- pyqtgraph
- scipy
- numpy


### Example
```
$ cp -r ~/.local/lib/python2.7/site-packages/crappy_crystals/examples .
$ crappy_crystals.py examples/example_duck_P1/config.ini
```

To display the ouput:
```
$ crappy_crystals.py -d examples/example_duck_P1/config.ini
```

### GPU example
If you have afnumpy: https://github.com/FilipeMaia/afnumpy
Then you may run this code on the gpu:
```
$ cp -r ~/.local/lib/python2.7/site-packages/crappy_crystals/examples .
$ crappy_crystals.py examples/example_duck_P1/config.ini
```

#### Getting afnumpy working on the CFEL maxwell machines
Do:
```
ssh -X max-cfelg
module load python/2.7
module load opencl/intel
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/amorgan/.local/lib/
```

