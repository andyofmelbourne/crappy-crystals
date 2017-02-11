# crappy-crystals
Simulate and phase 3D diffraction volumes for crystals with translational disorder.

### Installation as local user on Linux
```
$ git clone https://github.com/andyofmelbourne/crappy_crystals.git 
```
Done!

### Requires
- python (probably => 2.7)
- h5py 
- pyqtgraph (for the gui only)
- scipy
- numpy

### Example GUI
```
$ cd crappy_crystals
$ python gui/crappy_gui.py hdf5/duck/duck.h5
```
If hdf5/duck/duck.h5 does not exist (which it will not the first time) then **press Enter** when asked if you would like to create this file.

A Gui will then launch with some tabs up the top of the window. **Click on the 'Forward model' tab** then **click the 'Calculate forward model' button** in that tab. At the bottom of the window you should see the status of the calculation and the command that was executed (you can type this command into the command line yourself if you prefer not to use the gui). You should then see an image of the crystal density from three different directions and slices through the simulated diffraction volume of the crystal. Play with the parameters on the left to see what effect they have. 

Click the 'show h5 dataset' tab and then on the 'update' button to see the datasets that have been written to the file you have created. Click on a dataset and you will get a representation of it. 

**Now Click the 'Phase' tab** then the **'phase' button**. Wait, and... bam! you should see something that looks like the forward model (although the view of the crystal might be shifted if there are ambiguities). The 'iters' field on the left shows which projections where perfored in the phasing procedure and in what order. 

### Example command line 
```
$ cd crappy_crystals
$ python process/forward_model.py -f hdf5/duck/duck.h5 -c process/forward_model.ini
$ python process/phase.py -f hdf5/duck/duck.h5 -c process/phase.ini
```
Now you results are in the file 'hdf5/duck/duck.h5' in the field '/phase/solid_unit'.
