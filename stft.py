#!/usr/bin/env python3
"""
"Waterfall" frequency plot
"""

from numpy import abs, asarray, arange, hamming, log10
from numpy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from matplotlib.pylab import *

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('input')
    P.add_argument('-M','--mask',type=lambda v:int(v,0),
                   default=0xffffffff,
                   help='Channel bit mask')
    P.add_argument('-F', '--freq', type=float, default=250e3,
                   help='sampling frequency')
    P.add_argument('-W', '--window', type=int, default=10000,
                   help='window size in samples')
    return P

args = getargs().parse_args()

if args.input.endswith('.npz'):
    S = load(args.input)['adc'].astype('f8')
else:
    from h5py import File
    Sfile = File(args.input, 'r')
    S = Sfile['adc']
    args.freq = S.attrs['Fsamp']

#print('max', S.max(0))
TB = arange(S.shape[0])/args.freq

W = hamming(args.window)[:,None].repeat(S.shape[1], 1)
F = rfftfreq(args.window, 1/args.freq)

FFTs = []
ticks = []
idx = 0
while idx<S.shape[0]:
    #print("Q", idx, args.window)
    Sl = S[idx:idx+args.window, :]
    if Sl.shape[0]!=args.window:
        break

    T = rfft(Sl*W, axis=0)[None, ...]
    FFTs.append(T)
    ticks.append(TB[idx])

    idx += args.window//2 # slide...

T = np.concatenate(FFTs) # [#wind, #freq, #chan]

T = T[:, 1:, :] # discard F0
F = F[1:]

ticks = asarray(ticks)
extent = (
    F[0], # left
    F[-1],  # right
    ticks[-1], # top
    ticks[0], # bottom
)

for n in range(32):
    if not args.mask & 1<<n:
        continue
    #subplot(6, 6, n+1)
    figure()

#n=9
    title(f'ch {n}')
    imshow(log10(abs(T[:,:,15])), origin='upper', extent=extent, aspect='auto')
    ylabel(f'{ticks[1]} sec. per pixel')
    xlabel('Hz')
    xscale('log', base=10)

tight_layout()
show()
