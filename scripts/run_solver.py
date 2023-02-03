from pyMeDeCom import MeDeCom
import numpy as np
from os.path import join, dirname

if __name__ == "__main__":
    solver = MeDeCom(lmbda=0.1)
    D = np.load(join(dirname(__file__), "D.npy"))
    T, A, RMSE = solver.run_parallel(
        D=D, k=5, ninit=250, niter=50, ncores=2, progress=True
    )
    print(f"RMSE: {round(RMSE, 5)}")
