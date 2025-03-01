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

# avoid issue with type promotion to float64 during uint64 + int arithmatic
one = numpy.asarray(1, dtype='u8')

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('input', type=Path,
                   help='.dat file')
    P.add_argument('output', type=Path,
                   help='.dat file')
    return P

def main(args):
    print(args.input,'->',args.output)
    F = args.input.open('rb')
    body = F.read()

    H = numpy.frombuffer(body, dtype=_T)
    assert numpy.all(H['ps']==0x5053)

    D = numpy.diff(H['seq'])

    W, = numpy.where(D!=1)
    print('found', len(W), 'skips of', len(D))

    O = args.output.open('wb')

    i=0
    for w in W:
        # w   is index of last sample before drop
        # w+1 is index of first sample note dropped

        O.write(H[i:w+1].tobytes()) # [i, w+1)
        print('file offset', O.tell())

        t = H['sec'][w] + H['ns'][w]*1e-9
        print('jump at', t, time.ctime(t), end=', ')
        delt = H['seq'][w+1] - H['seq'][w]
        print(delt-1, 'packets missed', H['seq'][w], '->',  H['seq'][w+1])

        S = numpy.zeros(delt-one, dtype=_T)
        S['ps'] = H[0]['ps']
        S['msgid'] = H[0]['msgid']
        S['blen'] = H[0]['blen']
        S['chmask'] = H[0]['chmask']
        S['seq'] = numpy.arange(int(H['seq'][w])+1, int(H['seq'][w+1]), dtype='u8')
        S['samp'] = 0x7f
        print(H['seq'][w], S['seq'], H['seq'][w+1])

        O.write(S.tobytes()) # delt x placeholders

        i=w+1

    O.write(H[i:].tobytes())

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(getargs().parse_args())
