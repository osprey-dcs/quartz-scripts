#!/usr/bin/env python3

from scipy.optimize import curve_fit
from matplotlib.pylab import *

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('input')
    P.add_argument('freq', type=float)
    P.add_argument('-C', '--chan', type=int)
    return P

args = getargs().parse_args()

S = load(args.input)['adc']

for ch in range(S.shape[1]):
    print('input', ch, S[:,ch].min(), S[:,ch].mean(), S[:,ch].max())

T = arange(S.shape[0]) / 250e3
Vx = S[:,args.chan]


M0=Vx.max()
F0=args.freq
P0=pi/2

lower = (
    M0*0.95,
    F0*0.95,
    0,
)
upper = (
    M0*1.05,
    F0*1.05,
    2*pi,
)
Y0 = (M0, F0, P0)
print('lower', lower)
print('upper', upper)
print('initial', Y0)

def fit(X, M, F, pha):
    return M*sin(F*2*pi*X+pha)

Y, F, I, msg, ier =curve_fit(fit, T, S[:,0], p0=Y0, bounds=(lower, upper), full_output=True)
print(msg)

Vf = fit(T, *Y)

subplot(3,1,1)

plot(T, fit(T, *Y0), label='initial')
plot(T, Vf, label='fit')
plot(T, Vx, label='samp')
legend()
xlabel('s')
ylabel('ADC')
title(f'raw and fitted to {F0:.0f} Hz')

subplot(3,1,2)

Err=Vf-Vx
plot(T, abs(Err)) # /Vx.max()*100
xlabel('s')
ylabel('ADC')
title('abs(err)')

subplot(3,1,3)

hist(Err, 100)
title('histogram of err')

#print(I)
print('fit', Y)
print('err std', Err.std())

tight_layout()
show()
