from os.path import join, dirname

import numpy as np

from pyMeDeCom import MeDeCom

if __name__ == "__main__":
    solver = MeDeCom(lmbda=0.1)
    D = np.load(join(dirname(__file__), "D.npy"))
    T, A, RMSE = solver.run_parallel(
        D=D, k=5, ninit=250, niter=50, ncores=8, progress=True
    )
    print(f"RMSE: {round(RMSE, 5)}")
