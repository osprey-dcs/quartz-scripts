#!/usr/bin/env python3

from numpy import abs, angle, hamming, log10
from numpy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from matplotlib.pylab import *

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('input')
    P.add_argument('-F', '--freq', type=float, default=250e3,
                   help='sampling frequency')
    return P

args = getargs().parse_args()

S = load(args.input)['adc'].astype('f8')

Vpp = S.max(0) - S.min(0)
print('Vpp', Vpp)

S = S - S.mean(0)[None,:].repeat(S.shape[0], 0)

W = hamming(S.shape[0])[:,None].repeat(S.shape[1], 1)
T = rfft(S*W, axis=0)
F = rfftfreq(S.shape[0], 1/args.freq)
del W

A = abs(T)
I = 20*log10(A / T.max())
P = angle(T, deg=True)

subplot(2,1,1)

title(args.input)
plot(F, I)
ylabel('20*log10(abs(rfft(ADC)))')
xlabel('Hz')
#yscale('log')
grid(True)

subplot(2,1,2)

ChMask = A.max(0) < 0.3*A.max()
P[:, ChMask] = -180

plot(F, P)
ylabel('deg.')
ylim(-180, 180)
xlabel('Hz')
grid(True)

show()
