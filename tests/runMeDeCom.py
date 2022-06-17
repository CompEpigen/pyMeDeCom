""" Command line interface for pyMeDeCom

    Copyright(C) 2022

    Author: Valentin Maurer <valentin.maurer@stud.uni-heidelberg.de>

"""


from pyMeDeCom import MeDeCom
import numpy as np

if __name__ == "__main__":
    solver = MeDeCom()
    D = np.load("/Users/vale/src/pyMeDeCom/tests/D.npy")
    T, A, RMSE = solver.run_parallel(D = D, k = 5, ninit = 10, niter = 1000, ncores = 8,
        progress = True)
    print(RMSE)
