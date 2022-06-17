# pyMeDeCom

Python implementation of [MeDeCom's](https://github.com/lutsik/MeDeCom) cppTAfact.

## Configuration

The tool has been tested on python 3.9.13, but earlier versions should work just fine. pyMeDeCom and its dependencies can be installed as follows:

```
git clone git@github.com:CompEpigen/pyMeDeCom.git
cd pyMeDeCom
pip install .
```

## Running the software

An example on how to run pyMeDeCom can be found in ```tests/runMeDeCom.py```.

The ```run_parallel``` function starts the model fitting process. 
```k```corresponds to the number of latent components, ```ninit``` to the number of random initializations of the returned 
T and A matrices as in [MeDeCom](https://github.com/lutsik/MeDeCom), ```niter``` to the number of iterations per initialization and
```ncores``` to the number of cores that are used to fit the model.
