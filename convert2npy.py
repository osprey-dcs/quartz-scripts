#!/usr/bin/env python3

import sys
import os
import time
from pathlib import Path

import numpy

inp = sys.argv[1]
if len(sys.argv)>2:
    out = sys.argv[2]
else:
    out = str(Path(inp).with_suffix('.npz'))

R=Path(inp).read_bytes()
R=numpy.frombuffer(R, dtype='u1')
blen, = R[4:8].view('>u4')
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
    ('samp', 'u1', (blen - 4*6,)), # packed I24
])
P=R.view(_T)
assert numpy.all(P['ps']==0x5053)
assert numpy.all(P['msgid']==0x4e41)
assert numpy.all(P['blen']==blen)
assert numpy.all(P['chmask']==0xffffffff) # assume all channels enabled
Sb = P['samp'] # #pkt, #byte

Sb = Sb.reshape((Sb.shape[0], -1, 3)) # #samp, #pkt, 3

samp_per_pkt = Sb.shape[1] / 32

S = numpy.zeros(Sb.shape[:2]+(4,), dtype=Sb.dtype) # #samp, #pkt, 4
S[...,1:] = Sb
S[...,0 ] = numpy.bitwise_and(Sb[...,0], 0x80)/128*255 # sign extend

S = S.view('>i4').flatten() # concat packets
S = S.reshape((-1,32)) # #samp, 32

T = P['sec']+P['ns']*1e-9

print('T deltas', numpy.unique(numpy.diff(T)))

# sample frequency based on number of samples up to, but not
# including, first sample of last packet
Fsamp = (T.shape[0]-1)*samp_per_pkt/(T[-1]-T[0])

print('samples', S.shape)
print('T0', P['sec'][0], P['ns'][0], T[0], time.ctime(T[0]))
print('Fsamp', Fsamp)

with open(out, 'xb') as O:
    numpy.savez_compressed(O, adc=S, T=T, T0=T[0], Fsamp=Fsamp)
