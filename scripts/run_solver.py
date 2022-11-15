""" Command line interface for pyMeDeCom

    Copyright(C) 2022

    Author: Valentin Maurer <valentin.maurer@stud.uni-heidelberg.de>

"""


from pyMeDeCom import MeDeCom
import numpy as np
from os.path import join, dirname

if __name__ == "__main__":
    solver = MeDeCom()
    D = np.load(join(dirname(__file__), "D.npy"))
    T, A, RMSE = solver.run_parallel(D = D, k = 5, ninit = 10, niter = 100,
        ncores = 8, progress = True)
    print(f"RMSE: {round(RMSE, 5)}")
