# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 08:38:46 2019

@author: mjigmond
"""

import numpy as np
import os

def parseHDS(fn, nlay, nrow, ncol):
    """
    This function will parse a binary structured *.hds file
    and will return a tuple with two items.
    The first item is a dictionary containing the head arrays as values and
    stress periods/time steps tuples as keys.
    The second item is an array of total times matching
    the length of the dictionary.

    Parameters
    ----------
    fn : str
        Input HDSU file name.
    nlay : int
        Number of layers in the model.
    nrow : int
        Number of rows in the model.
    ncol : int
        Number of columns in the model.

    Returns
    -------
    tuple
        The first item is a dictionary containing the head arrays as values and
        stress periods/time steps tuples as keys.
        The second item is an array of total times matching
        the length of the dictionary.

    """
    # """
    # This function will parse a structured binary *.hds file
    # and will return a tuple with two items.
    # The first item is dictionary containing the head arrays as values and
    # stress periods/time steps tuples as keys.
    # The second item is an array of total times matching
    # the length of the dictionary.
    # """
    if fn is None or nlay == 0 or nrow == 0 or ncol == 0:
        return ({}, np.empty(0))
    fsize = os.path.getsize(fn)
    ofst = 0
    hds = {}
    totim = []
    while ofst < fsize:
        dt = np.dtype([
                ('kstp', 'i4'),
                ('kper', 'i4'),
                ('pertim', 'f4'),
                ('totim', 'f4'),
                ('text', 'S16'),
                ('ncol', 'i4'),
                ('nrow', 'i4'),
                ('ilay', 'i4'),
                ('data', 'f4', (nrow, ncol))
        ])
        arr = np.memmap(fn, mode='r', dtype=dt, offset=ofst, shape=nlay)
        ofst += dt.itemsize * nlay
        kper = arr['kper'][0]
        kstp = arr['kstp'][0]
        hds[kper, kstp] = arr['data'].copy()
        totim.append(arr['totim'][0])
    return hds, np.asarray(totim)

def parseCBB(fn, nlay, nrow, ncol, items=[]):
    """
    This function will parse a binary structured *.cbb file
    and will return a dictionary.

    Parameters
    ----------
    fn : str
        DESCRIPTION.
    nlay : int
        Number of layers in the model.
    nrow : int
        Number of rows in the model.
    ncol : int
        Number of columns in the model.
    items: list
        List of budget items to return. Large budget files can excceed
        available memory so limit the returned items.

    Returns
    -------
    dict
        containing the flux arrays as values and
        stress periods/time steps tuples as keys. Budget terms are stripped of
        white space at the beginning and at the end.
    """
    fsize = os.path.getsize(fn)
    ofst = 0
    cbb = {}
    while ofst < fsize:
        dt = np.dtype([
                ('kstp', 'i4'),
                ('kper', 'i4'),
                ('text', 'S16'),
                ('ncol', 'i4'),
                ('nrow', 'i4'),
                ('nlay', 'i4'),
                ('data', 'f4', (nlay, nrow, ncol))
        ])
        arr = np.memmap(fn, mode='r', dtype=dt, offset=ofst, shape=1)[0]
        ofst += dt.itemsize
        text = arr['text'].decode().strip()
        kper = arr['kper']
        kstp = arr['kstp']
        if text in items:
            cbb.setdefault(text, {})
            cbb[text][kper, kstp] = arr['data'].copy()
    return cbb

def parseHDSu(fn):
    """
    Parses a binary unstructered heads file.
    
    Parameters
    ----------
    fn : str
        Input HDSU file name.

    Returns
    -------
    dict
        Dictionary of the following type:
            {(kper: int, kstp: int): np.array}.

    """
    fSize = os.path.getsize(fn)
    print(fSize)
    ofst = 0
    HDS = {}
    while ofst < fSize:
        dtmeta = np.dtype([
                ('kstp', 'i4'), 
                ('kper', 'i4'),
                ('pertim', 'f4'),
                ('totim', 'f4'),
                ('text', 'S16'),
                ('nstrt', 'i4'),
                ('nndlay', 'i4'),
                ('ilay', 'i4')
        ])
        meta = np.memmap(fn, mode='r', dtype=dtmeta, offset=ofst, shape=1)[0]
        print('debug message', meta)
        nstrt = meta['nstrt']
        nndlay = meta['nndlay']
        kper = meta['kper']
        kstp = meta['kstp']
        ilay = meta['ilay']
        if ilay == 1:
            h = np.empty(0)
        n = nndlay - nstrt + 1
        dt = np.dtype([
                ('kstp', 'i4'), 
                ('kper', 'i4'), 
                ('pertim', 'f4'),
                ('totim', 'f4'), 
                ('text', 'S16'),
                ('nstrt', 'i4'), 
                ('nndlay', 'i4'),
                ('ilay', 'i4'),
                ('data', 'f4', n)
        ])
        arr = np.memmap(fn, mode='r', dtype=dt, offset=ofst, shape=1)[0]
        h = np.concatenate((h, arr['data']))
        ofst += dt.itemsize
        HDS[kper, kstp] = h
    return HDS

def parseCBBu(fn, items=[]):
    """
    Parses a binary unstructured budget file.

    Parameters
    ----------
    fn : str
        Input CBBU file name.
    items : list
        List of budget items to return. Large budget files can excceed
        available memory so limit the returned items.

    Returns
    -------
    dict
        Dictionary of the following type:
            {((item: str, array_size: int), kper: int, kstp: int): np.array}.

    """
    fSize = os.path.getsize(fn)
    print(fSize)
    ofst = 0
    BUD = {}
    while ofst < fSize:
        dtmeta = np.dtype([
                ('kstp', 'i4'),
                ('kper', 'i4'),
                ('text', 'S16'),
                ('nval', 'i4'),
                ('one', 'i4'),
                ('icode', 'i4')
        ])
        meta = np.memmap(fn, mode='r', dtype=dtmeta, offset=ofst, shape=1)[0]
        arrSize = meta['nval']
        print('debug message', meta)
        dtu = np.dtype([
                ('kstp', 'i4'),
                ('kper', 'i4'), 
                ('text', 'S16'),
                ('nval', 'i4'), 
                ('one', 'i4'),
                ('icode', 'i4'), 
                ('data', 'f4', arrSize)
        ])
        data = np.memmap(fn, mode='r', dtype=dtu, offset=ofst, shape=1)[0]
        ofst += dtu.itemsize
        print('debug message', ofst, fSize, fSize - ofst)
        kper = data['kper']
        kstp = data['kstp']
        text = data['text'].decode().strip()
        if text in items:
            BUD[(text, arrSize), kper, kstp] = data['data']
    return BUD

if __name__ == '__main__':
    nlay, nrow, ncol = 7, 368, 410
    fName = 'data/abr.hds'
    hds = parseHDS(fName, nlay, nrow, ncol)
    
    fName = 'data/abr.cbb'
    cbb = parseCBB(fName, nlay, nrow, ncol, items=['ET', 'WELLS', 'RIVER LEAKAGE'])
    print(cbb.keys())

    fName = 'data/biscayne.hds'
    hds = parseHDSu(fName)
    
    fName = 'data/biscayne.cbc'
    cbb = parseCBBu(fName, items=['ET', 'WELLS', 'RIVER LEAKAGE'])
    print(cbb.keys())
