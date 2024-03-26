#!/usr/bin/env python3
import sys
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
TB = arange(S.shape[0])/args.freq

plot(TB, S, label=[str(n) for n in range(S.shape[1])])
ylabel('ADC')
xlabel('s')
legend()
show()
