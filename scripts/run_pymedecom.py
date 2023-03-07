#python3

import argparse
import pickle
from itertools import product

import numpy as np

from pyMeDeCom import MeDeCom

# https://stackoverflow.com/questions/14117415/in-python-using-argparse-allow-only-positive-integers
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve factorization problem D = T @ A for T and A,"
        " given ground truth T and A."
    )
    parser.add_argument(
        "-t",
        dest="exposure",
        type=str,
        required=True,
        help="Path to ground truth T matrix.",
    ),
    parser.add_argument(
        "-a",
        dest="proportion",
        type=str,
        required=True,
        help="Path to ground truth A matrix.",
    )
    parser.add_argument(
        "-tout",
        dest="exposure_out",
        type=str,
        required=True,
        help="Path to write fitted T matrix to.",
    ),
    parser.add_argument(
        "-aout",
        dest="proportion_out",
        type=str,
        required=True,
        help="Path to write fitted A matrix to.",
    )
    parser.add_argument(
       "-out",
       dest="outpath",
       type=str,
       required=False,
       default = None,
       help="Path to write dictionary to.",
    ),
    parser.add_argument(
        "-n",
        dest="cores",
        required=False,
        default=4,
        type=check_positive,
        help="Number of cores for fitting.",
    ),
    parser.add_argument(
        "-l",
        dest="lambdas",
        required=False,
        default="0.01",
        type=str,
        help="Comma separated sequence of lambdas, e.g. '0.1,1,10'",
    ),
    parser.add_argument(
        "-k",
        dest="components",
        required=False,
        default="5",
        type=str,
        help="Comma separated sequence of ks, e.g. '1,2,3'",
    ),
    parser.add_argument(
        "-n",
        dest="ninit",
        required=False,
        default=5,
        type=int,
        help="Number of random initializations.",
    ),
    parser.add_argument(
        "-ni",
        dest="niter",
        required=False,
        default=5,
        type=int,
        help="Maximum number of iterations per factorization.'",
    ),
    parser.add_argument(
        "-d",
        dest="delimiter",
        type=str,
        required=False,
        default=",",
        help="Delimiter for input and output matrices.",
    )
    args = parser.parse_args()

    args.components = [int(element) for element in args.components.split(',')]
    args.lambdas = [float(element) for element in args.lambdas.split(',')]

    return args


def load_matrix(
    filename: str, default_dtype: np.dtype = np.float64, delimiter: str = None
) -> np.ndarray:
    if filename.endswith("npy"):
        return np.load(filename).astype(default_dtype)
    return np.loadtxt(filename, delimiter=delimiter, dtype=default_dtype)

def save_pickle(path_save: str, data: dict) -> None :
    if ".pickle" not in path_save:
        path_save = path_save + ".pickle"
    with open(path_save, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename: str) -> dict:
    with open(filename, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

if __name__ == "__main__":
    args = parse_args()

    T = load_matrix(args.exposure, delimiter=args.delimiter)
    A = load_matrix(args.proportion, delimiter=args.delimiter)

    D = T @ A
    components = A.shape[0]
    dictionary = {}
    for k, lmbda in product(args.components, args.lambdas)
        solver = MeDeCom(lmbda=lmbda)
        T, A, RMSE = solver.run_parallel(
            D=D,
            k=k,
            ninit=args.ninit,
            niter=args.niter,
            ncores=args.cores,
            progress=True,
        )
        print(f"RMSE: {round(RMSE, 5)} [k={k} lambda={lmbda}]")
        file_name = f"k{k}_l{lmbda}"
        dictionary[file_name] = (T, A, RMSE)

    if args.outpath is not None:
        save_pickle(args.outpath, dictionary)

    # Save based on top hit RMSE
    min_rmse = 1000
    name_min = ''
    for key, item in dictionary.items():
        if item[2] < min_rmse :
            min_rmse = item[2]
            name_min = key
    print("Min RMSE ", min_rmse, " for ", name_min)
    T = (dictionary[name_min])[0]
    A = (dictionary[name_min])[1]
    
    np.savetxt(args.exposure_out, T, delimiter=args.delimiter)
    np.savetxt(args.proportion_out, A, delimiter=args.delimiter)
