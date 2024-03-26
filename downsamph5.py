#!/usr/bin/env python3

import sys
import os
import time
import struct
import logging
from pathlib import Path

import numpy
import h5py
from scipy.signal import decimate

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('--div', type=int, default=10,
                   help='Divider on sample frequency')
    P.add_argument('input', help='Input HDF5')
    P.add_argument('output', help='Output HDF5')
    return P

def main(args):
    with h5py.File(args.input, 'r') as IN, h5py.File(args.output, 'w') as OUT:
        S = IN['adc']

        Fsamp = S.attrs['Fsamp']

        Fsamp /= args.div
        Sd = decimate(S, args.div, ftype='fir', axis=0)

        O = OUT.create_dataset('adc', data=Sd)

        # copy all attributes verbatim
        for K, V in S.attrs.items():
            O.attrs[K] = V

        # overwrite some
        O.attrs['Fsamp'] = Fsamp

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(getargs().parse_args())
