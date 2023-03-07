from os.path import join, dirname

import os
import numpy as np

from pyMeDeCom import MeDeCom

if __name__ == "__main__":
    path_outs = "/mnt/c/Users/r0962324/pyMeDeCom/scripts/outs"
    lambdas = [0.01, 0.1]
    components = [1,5]
    for k in components :
        for l in lambdas :
            solver = MeDeCom(lmbda=l)
            D = np.load(join(dirname(__file__), "D.npy"))
            T, A, RMSE = solver.run_parallel(
                D=D, k=k, ninit=250, niter=50, ncores=8, progress=True
            )
            print("for k=", k, " and lambda=", l, " : ")
            print(f"RMSE: {round(RMSE, 5)}")

            if not (os.path.exists(path_outs)):
                    os.makedirs(path_outs)
            exposure_outs = os.path.join(path_outs, "exposure_t")
            if not (os.path.exists(exposure_outs)):
                os.makedirs(exposure_outs)
            proportion_outs = os.path.join(path_outs, "proportion_a")
            if not (os.path.exists(proportion_outs)):
                os.makedirs(proportion_outs)

            file_name = "k"+str(k)+"lambda"+str(l)
            exposure_outs = os.path.join(exposure_outs, file_name)
            if not (os.path.exists(exposure_outs)):
                os.makedirs(exposure_outs)
            proportion_outs = os.path.join(proportion_outs,file_name)
            if not (os.path.exists(proportion_outs)):
                os.makedirs(proportion_outs)
            
            np.savetxt(os.path.join(exposure_outs, "T.txt"), T)
            np.savetxt(os.path.join(proportion_outs, "A.txt"), A)
