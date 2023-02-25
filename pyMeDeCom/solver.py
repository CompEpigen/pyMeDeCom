""" Python implementation of MeDeCom https://github.com/lutsik/MeDeCom

    Copyright (C) 2022

    Author: Valentin Maurer <valentin.maurer@stud.uni-heidelberg.de>

"""

import random
import sys
import traceback
from copy import deepcopy

import numpy as np
from sklearn.utils import check_random_state
from tqdm import tqdm
from time import sleep
from multiprocessing import RawValue, Lock, Process, shared_memory, Queue
from .extensions import TAFact


class _Counter:
    def __init__(self):
        self.val = RawValue("i", 0)
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
        self._relerror = RawValue("d", 10**12)
        self._dtype = np.float64
        self._exceptions = Queue()

        if not hasattr(self, "arr_order"):
            self.arr_order = "C"

    @staticmethod
    def arr_to_sharedarr(arr: np.ndarray):
        """
        Converts an numpy array to an object shared in memory
        """
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        np_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np_array[:] = arr[:]
        return shm

    def sharedarr_to_arr(self, shape, dtype, shm):
        return np.ndarray(
            shape=shape, dtype=dtype, buffer=shm.buf, order=self.arr_order
        )

    @staticmethod
    def _free_sharedarr(link):
        link.close()
        link.unlink()

    def _update_matrices(self, T: np.ndarray, A: np.ndarray, error: float):
        m, k = T.shape
        n = A.shape[1]
        with Lock():
            if error < self._relerror.value:
                _T = self.sharedarr_to_arr((m, k), self._dtype, self._T)
                _A = self.sharedarr_to_arr((k, n), self._dtype, self._A)
                self._relerror.value = error
                _T[:] = np.asarray(T, order=self.arr_order)
                _A[:] = np.asarray(A, order=self.arr_order)
        self._counter.increment()

    @staticmethod
    def _handle_traceback(last_type, last_value, last_traceback):
        if last_type is None:
            return None
        traceback.print_tb(last_traceback)
        raise last_type(last_value)

    def clean_shared_memory(self):
        for attribute in self.__dict__:
            class_attribute = getattr(self, attribute)
            if class_attribute.__class__ is not shared_memory.SharedMemory:
                continue
            try:
                self._free_sharedarr(class_attribute)
                setattr(self, attribute, None)
            # Shared memory has been freed already
            except FileNotFoundError:
                continue

    def _shared_memory_handler(func):
        def inner_function(self, *args, **kwargs):
            last_type, last_value, last_traceback = sys.exc_info()
            try:
                return func(self, *args, **kwargs)
            except Exception:
                last_type, last_value, last_traceback = sys.exc_info()
            # Free shared memory
            finally:
                self.clean_shared_memory()
                self._handle_traceback(last_type, last_value, last_traceback)

        return inner_function

    def _print_progress(self, progress: bool, ninit: int) -> None:
        if not progress:
            return None
        with tqdm(total=ninit) as pbar:
            last_check = 0
            pbar.set_description("Factorization")
            while last_check < ninit:
                self._supervise_processes()
                pbar.update(self._counter.value() - last_check)
                last_check = self._counter.value()
                sleep(0.2)
            if pbar.n < ninit:
                pbar.update(ninit - pbar.n)

    def _supervise_processes(self) -> None:
        while not self._exceptions.empty():
            callback = self._exceptions.get()
            if callback is None:
                continue
            raise callback[0](callback[1])

    @_shared_memory_handler
    def run_parallel(
        self,
        D: np.ndarray,
        k: int,
        ninit: int,
        niter: int,
        ncores: int = 1,
        progress: bool = True,
        seed=None,
    ):

        # Reset variables
        self._counter.reset()

        # Calculate unique seed for each process based on supplied seed
        random.seed(seed)
        seedlist = random.sample(range(0, 2 << 32 - 1), k=ncores)
        rngs = [check_random_state(seed) for seed in seedlist]

        D = np.asarray(D, order=self.arr_order)

        self._T = self.arr_to_sharedarr(
            np.zeros((D.shape[0], k), dtype=self._dtype, order=self.arr_order)
        )
        self._A = self.arr_to_sharedarr(
            np.zeros((k, D.shape[1]), dtype=self._dtype, order=self.arr_order)
        )

        jobs = []
        runs_per_process = int(np.ceil(ninit / ncores))
        for n in range(ncores):
            jobs.append(
                Process(
                    target=self.run_repeat,
                    args=(D, k, runs_per_process, niter, rngs[n]),
                )
            )

        [jobs[n].start() for n in range(ncores)]
        self._print_progress(progress, ninit)
        [jobs[n].join() for n in range(ncores)]
        self._supervise_processes()

        # Copy sink into output array and free shared memory
        T = deepcopy(self.sharedarr_to_arr((D.shape[0], k), self._dtype, self._T))
        A = deepcopy(self.sharedarr_to_arr((k, D.shape[1]), self._dtype, self._A))

        # RMSE definition used in MeDeCom
        rmse = np.linalg.norm(0.5 * (D - T @ A)) ** 2
        rmse /= D.shape[0] * D.shape[1]

        return T, A, rmse


class MeDeCom(Base_Solver):
    def __init__(self, lmbda: float = 0):
        super().__init__()
        self.tol = 1e-8
        self.tolA = 1e-7
        self.tolT = self.tolA
        self.lmbda = lmbda
        self.arr_order = "F"

    @staticmethod
    def _initialize(D: np.ndarray, k: int, rng: np.random.mtrand.RandomState):

        m, n = D.shape

        Tt = rng.rand(k, m)
        A = -np.log(rng.rand(k, n))
        A /= A.sum(axis=0)[None, :]

        Tt = np.asarray(Tt, order="F")
        A = np.asarray(A, order="F")

        return Tt, A

    def run_repeat(
        self,
        D: np.ndarray,
        k: int,
        ninit: int,
        niter: int,
        rng: np.random.mtrand.RandomState,
    ):
        try:
            self._run_repeat(D=D, k=k, ninit=ninit, niter=niter, rng=rng)
        except Exception:
            last_type, last_value, last_traceback = sys.exc_info()
            self._exceptions.put((last_type, last_value))

    def _run_repeat(
        self,
        D: np.ndarray,
        k: int,
        ninit: int,
        niter: int,
        rng: np.random.mtrand.RandomState,
    ):
        Dt = np.asarray(D.T, order="F")
        for _ in range(ninit):
            Tt, A = self._initialize(D=D, k=k, rng=rng)
            rmse = TAFact(
                D=Dt,
                Tt0=Tt,
                A0=A,
                lmbda=self.lmbda,
                itersMax=niter,
                innerItersMax=niter,
                tol=1e-8,
                tolT=1e-7,
                tolA=1e-7,
            )
            self._update_matrices(Tt.T, A, rmse.rmse)
