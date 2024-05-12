#!/usr/bin/env python3

import sys
import os
import time
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy
import h5py

_log = logging.getLogger(__name__)

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('csv', type=Path)
    P.add_argument('outdir', type=Path)
    return P

def main(args):
    args.outdir.mkdir(parents=True, exist_ok=True)

    Outfile = {}
    with args.csv.open('r') as CSV:
        for R in csv.DictReader(CSV):
            use = R['USE'].lower()
            assert use in ('yes', 'no'), use
            if use!='yes':
                continue

            cnum = int(R['CHASSIS'])
            if cnum not in Outfile:
                Outfile[cnum] = (args.outdir / f'CH{cnum:02d}-scale.txt').open('w')

            OF = Outfile[cnum]

            # write: chan, offset, slope, unit, name
            #  1, 0.0, 11834.3195266272, Pascal, CM1

            OF.write(f"{R['CHANNEL']}, {R['EOFF']}, {R['ESLO']}, {R['EGU'].strip()}, {R['CUSTNAM'].strip()}\n")

    pass

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(getargs().parse_args())
