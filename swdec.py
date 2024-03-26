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

IN=h5py.File('x.h5', 'r')
OUT=h5py.File('y.h5', 'w')

I=IN['adc']

Q=decimate(I, 10, ftype='fir', axis=0)

O=OUT.create_dataset('adc', data=Q)

for attr in ('units',
             'labels',
             'calib',
             'scale',
             'T0',
             'T0_desc',
            ):
    O.attrs[attr] = I.attrs[attr]
