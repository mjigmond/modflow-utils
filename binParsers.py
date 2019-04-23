# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:38:46 2019

@author: mjigmond
"""

import numpy as np
import os

def parseHDS(fn=None, nlay=0, nrow=0, ncol=0) -> tuple:
    '''This function will parse a binary *.hds file
    and will return a tuple with two items.
    The first item is dictionary containing the head arrays as values and 
    stress periods/time steps tuples as keys.
    The second item is an array of total times matching the length of the dictionary.
    '''
    if fn is None or nlay == 0 or nrow == 0 or ncol == 0:
        return {}
    fsize = os.path.getsize(fn)
    ofst = 0
    hds = {}
    totim = []
    while ofst < fsize:
        dt = np.dtype([
                ('kstp', 'i4')
                ,('kper', 'i4')
                ,('pertim', 'f4')
                ,('totim', 'f4')
                ,('desc', 'S16')
                ,('ncol', 'i4')
                ,('nrow', 'i4')
                ,('ilay', 'i4')
                ,('data', 'f4', (nrow, ncol))
        ])
        arr = np.memmap(fn, mode='r', dtype=dt, offset=ofst, shape=nlay)
        ofst += dt.itemsize * nlay
        kper = arr['kper'][0]
        kstp = arr['kstp'][0]
        hds[kper, kstp] = arr['data'].copy()
        totim.append(arr['totim'][0])
    return hds, np.asarray(totim)