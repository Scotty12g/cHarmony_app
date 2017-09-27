#!/usr/bin/env python
import pickle

class Pickler:

    @staticmethod
    def save_pickle(obj, binhandle):
        pickle.dump(obj, binhandle)

    @staticmethod
    def load_pickle(binhandle):
        return pickle.load(binhandle)