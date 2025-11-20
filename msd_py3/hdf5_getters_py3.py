# hdf5_getters_py3.py
from hdf5_getters import *
import tables

def open_h5_file_read(h5filename):
    return tables.open_file(h5filename, mode='r')

def open_h5_file_append(h5filename):
    return tables.open_file(h5filename, mode='a')
