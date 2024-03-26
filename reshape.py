from pathlib import Path
R=Path('FOO-cw.dat').read_bytes()
R=numpy.frombuffer(R, dtype='u1')
blen, = R[4:8].view('>u4')
_T = numpy.dtype([
    ('ps', '>u2'),
    ('msgid', '>u2'),
    ('blen', '>u4'),
    ('rsec', '>u4'),
    ('rns', '>u4'),
    ('info', 'u4', (6,)),
    ('samp', 'u1', (blen - 4*6,)),
])
P=R.view(_T)
assert np.all(P['ps']==0x5053)
assert np.all(P['msgid']==0x4e41)
assert np.all(P['blen']==blen)
assert np.all(P['info'][:,1]==0xffffffff)
Sb = P['samp'] # #pkt, #byte

Sb = Sb.reshape((Sb.shape[0], -1, 3)) # #samp, #pkt, 3

S = numpy.zeros(Sb.shape[:2]+(4,), dtype=Sb.dtype)
S[...,1:] = Sb
S[...,0 ] = numpy.bitwise_and(Sb[...,0], 0x80)/128*255

S = S.view('>i4').flatten()
S = S.reshape((-1,32))

#S = S.astype('f8') / S.max()

W = numpy.hamming(S.shape[0])[:,None].repeat(32, 1)
T = numpy.fft.rfft(S*W, axis=0)
F = numpy.fft.rfftfreq(S.shape[0], 1/250e3)

plot(F, numpy.log10(T))
