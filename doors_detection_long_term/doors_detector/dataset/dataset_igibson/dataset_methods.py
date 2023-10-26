import numpy as np
import gzip
import json

from typing import NoReturn

def save_tuple_list(path: str, data: list) -> NoReturn:
    with open(path + '.txt', mode='w') as file:
        file.write(str(data))

def load_tuple_list(path: str) -> list:
    with open(path + '.txt', mode='r') as file:
        return eval(file.readline())

def load_compressed_numpy_array(path: str) -> np.ndarray:
    with gzip.open(path + '.tar.gz', mode='r') as file:
        return np.load(file, allow_pickle=True)