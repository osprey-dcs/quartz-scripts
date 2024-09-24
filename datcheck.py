#!/usr/bin/env python3

import struct
import time
import logging
from pathlib import Path
import numpy

_log = logging.getLogger(__name__)

_psc_body = struct.Struct('>HHIII')

_T = numpy.dtype([
    ('ps', '>u2'),
    ('msgid', '>u2'),
    ('blen', '>u4'),
    ('rsec', '>u4'),
    ('rns', '>u4'),
    ('sts', '>u4'),
    ('chmask', '>u4'),
    ('seq', '>u8'),
    ('sec', '>u4'),
    ('ns', '>u4'),
    ('hihi', '>u4'),
    ('hi', '>u4'),
    ('lo', '>u4'),
    ('lolo', '>u4'),
    ('samp', 'u1', (14*3*32,)), # packed I24
])

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('dats', type=Path,
                   nargs='+', default=[],
                   help='.dat file')
    return P

def show(dat: Path):
    print(dat)
    F = dat.open('rb')
    body = F.read()

    H = numpy.frombuffer(body, dtype=_T)
    assert numpy.all(H['ps']==0x5053)

    D = numpy.diff(H['seq'])

    W, = numpy.where(D!=1)
    print('found', len(W), 'skips of', len(D))

    for w in W:
        t = H['sec'][w] + H['ns'][w]*1e-9
        print('jump at', t, time.ctime(t), end=', ')
        delt = H['seq'][w+1] - H['seq'][w]
        print(delt, 'packets missed')


def main(args):
    for dat in args.dats:
        show(dat)

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(getargs().parse_args())
