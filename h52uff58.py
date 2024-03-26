#!/usr/bin/env python3

import sys
import os
import time
import struct
import logging
from pathlib import Path

import numpy
import h5py

_log = logging.getLogger(__name__)

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('-M','--mask',type=lambda v:int(v,0),
                   default=0xffffffff,
                   help='Channel bit mask')
    P.add_argument('h5', help='Input HDF5')
    P.add_argument('uff', help='Output UFF58b')
    return P

def main(args):
    with h5py.File(args.h5, 'r') as IN, open(args.uff, 'wb') as OUT:
        S = IN['adc']
        _log.info('Input shape %r', S.shape)
        assert S.ndim==2, S.shape
        units = S.attrs['units']
        labels = S.attrs['labels']
        assert units.shape==(S.shape[1],)
        assert labels.shape==(S.shape[1],)

        T0 = S.attrs['T0']
        Tzero = time.strftime('%d-%b-%y %H:%M:%S', time.gmtime(T0))

        Tsamp = 1/250e3
        Tsamp *= 10

        idnum = 700 # TODO: better choices?
        bytes_per_channel = S.nbytes//S.shape[1]
        for ch in range(S.shape[1]):
            if not (args.mask & (1<<ch)):
                _log.debug('Skip channel %d', ch)
                continue
            _log.info('channel %d', ch)
            EGU = units[ch]
            DESC = labels[ch]
            fmt = 1 # f32 real
            endian = S.dtype.byteorder
            if endian=='=':
                endian = '<' if sys.byteorder == 'little' else '>'
            endian = {'<':1, '>':2}[endian]

            OUT.write(f'    -1\n    58b{endian: 6}     2          11{bytes_per_channel: 12}     0     0           0           0\n'.encode())
            # records 1-5
            OUT.write(f'{DESC[:79]}\nNONE\n{Tzero}\nNONE\nNONE\n'.encode())
            # record 6
            # TODO: direction
            OUT.write(f'    1         0    0         0 {EGU[:10]: <10}{idnum+ch: 10d}   0 NONE      {idnum+ch: 10d}   0\n'.encode())
            # record 7
            OUT.write(f'         2{S.shape[0]: 10d}         1 0.00000E+000{Tsamp:13.5e} 0.00000E+000\n'.encode())
            # record 8
            OUT.write('        17    0    0    0 Time                 s                   \n'.encode())
            # record 9
            OUT.write(f'         1    0    0    0 {DESC[:20]: <20} {EGU[:20]: <20}\n'.encode())
            # record 10
            OUT.write('         0    0    0    0 NONE                 NONE                \n'.encode())
            # record 11
            OUT.write('         0    0    0    0 NONE                 NONE                \n'.encode())
            # data...
            B0 = OUT.tell()
            if S.chunks is not None:
                for SL in S.iter_chunks((
                    slice(0, S.shape[0]),
                    slice(ch, ch+1),
                )):
                    _log.debug('slice %r', SL)
                    OUT.write(S[SL])
            else:
                OUT.write(S[:,ch])
            B1 = OUT.tell()
            assert bytes_per_channel==(B1-B0)
            OUT.write(b'    -1\n')


if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(getargs().parse_args())
