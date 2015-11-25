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

### Based on the paper
- Awesome Nature paper yet to be published...
