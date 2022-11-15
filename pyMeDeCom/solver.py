""" Python implementation of MeDeCom https://github.com/lutsik/MeDeCom

    Copyright (C) 2022

    Author: Valentin Maurer <valentin.maurer@stud.uni-heidelberg.de>

"""

import random
from copy import deepcopy

import numpy as np
from sklearn.utils import check_random_state
from tqdm import tqdm
from time import time, sleep
from multiprocessing import RawValue, \
                            Lock, \
                            Process, \
                            Manager, \
                            shared_memory, \
                            current_process


from .extensions import TAFact



class _Counter():

    def __init__(self):
        self.val = RawValue('i', 0)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def reset(self):
        with self.lock:
            self.val.value = 0

    def value(self):
        with self.lock:
            return self.val.value

class Base_Solver:
    """
    Function:
    Solve D = T x A by fitting T and A through non-negative matrix factorization

    Input:
    D[m x n] := m methylation sites measured from n patients

    Output:
    T[m x g] := translation between m methylation sites and k celltypes
    A[k x n] := proportion of k cell types in n patients
    """


    def __init__(self):
        self._error = 1
        self._counter = _Counter()
        self._sharedarrlock = Lock()
        self._relerror = RawValue('d', 10 ** 12)
        self._dtype = np.float64

        if not hasattr(self, "arr_order"):
            self.arr_order = "C"


    @staticmethod
    def arr_to_sharedarr(arr):
        '''
        Converts an numpy array to an object shared in memory
        '''
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        np_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np_array[:] = arr[:]
        return shm

    def sharedarr_to_arr(self, shape, dtype, shm):
        return np.ndarray(
            shape = shape, dtype = dtype,
            buffer = shm.buf, order = self.arr_order
        )

    @staticmethod
    def _free_sharedarr(link):
        link.close()
        link.unlink()

    def _update_matrices(self, T, A, error):
        m, k = T.shape
        n = A.shape[1]
        with Lock():
            if error < self._relerror.value:
                _T = self.sharedarr_to_arr((m, k),
                    self._dtype, self._T)
                _A = self.sharedarr_to_arr((k, n),
                    self._dtype, self._A)
                self._relerror.value = error
                _T[:] = np.asarray(T, order = self.arr_order)
                _A[:] = np.asarray(A, order = self.arr_order)
        self._counter.increment()


    def run_parallel(self, D, k, ninit, niter, ncores = 1, progress = True,
        seed = None):

        # Reset variables
        self._counter.reset()

        # Calculate unique seed for each process based on supplied seed
        random.seed(seed)
        seedlist = random.sample(range(0, 2<<32 -1), k = ncores)
        rngs = [check_random_state(seed) for seed in seedlist]

        D = np.asarray(D, order = self.arr_order)

        self._T = self.arr_to_sharedarr(
            np.zeros(
                (D.shape[0], k), dtype = self._dtype, order = self.arr_order
            )
        )
        self._A = self.arr_to_sharedarr(
            np.zeros(
                (k, D.shape[1]), dtype = self._dtype, order = self.arr_order
            )
        )

        jobs = []
        runs_per_process = int(np.ceil(ninit / ncores))
        for n in range(ncores):
            jobs.append(
                Process(
                    target = self.run_repeat,
                    args = (D, k, runs_per_process, niter, rngs[n])
                    )
                )

        [jobs[n].start() for n in range(ncores)]

        if progress:
            with tqdm(total = ninit) as pbar:
                last_check = 0
                pbar.set_description("Factorization")
                while last_check < ninit:
                    pbar.update(self._counter.value() - last_check)
                    last_check = self._counter.value()
                    sleep(.2)
                if pbar.n < ninit:
                    pbar.update(ninit - pbar.n)

        [jobs[n].join() for n in range(ncores)]

        # Copy sink into output array and free shared memory
        T = deepcopy(
            self.sharedarr_to_arr((D.shape[0], k), self._dtype, self._T)
        )
        A = deepcopy(
            self.sharedarr_to_arr((k, D.shape[1]), self._dtype, self._A)
        )
        self._free_sharedarr(self._T)
        self._free_sharedarr(self._A)

        # RMSE definition used in MeDeCom
        rmse = np.linalg.norm(0.5 * (D - T @ A)) ** 2
        rmse /= (D.shape[0] * D.shape[1])

        return T, A, rmse

class MeDeCom(Base_Solver):

    def __init__(self, lmbd = 0):
        super().__init__()
        self.tol  = 1e-8
        self.tolA = 1e-7
        self.tolT = self.tolA
        self.lmbd = lmbd
        self.arr_order = "F"

    @staticmethod
    def _initialize(D, k, rng):

        m,n = D.shape

        Tt = rng.rand(k, m)
        A = - np.log(rng.rand(k, n))
        A /= A.sum(axis = 0)[None, :]

        Tt = np.asarray(Tt, order = "F")
        A = np.asarray(A, order = "F")

        return Tt, A

    def run_repeat(self, D, k, ninit, niter, rng):

        Dt = np.asarray(D.T, order = "F")
        for _ in range(ninit):
            Tt, A = self._initialize(D = D, k = k, rng = rng)
            rmse = TAFact(
                D = Dt, Tt0 = Tt , A0 = A,
                lmbda = 0.0, itersMax = niter, innerItersMax = niter,
                tol = 1e-8, tolT = 1e-7, tolA = 1e-7
            )
            self._update_matrices(Tt.T, A, rmse.rmse)
