#!/usr/bin/env python3

import sys
import os
import time
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy
import h5py

_log = logging.getLogger(__name__)

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('json', type=Path)
    P.add_argument('outdir', type=Path)
    return P

def main(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    with args.json.open('r') as F:
        info = json.load(F)

    prefix = info['AcquisitionId']

    Outfile = {}
    for sig in info['Signals']:
        cnum = sig['Address']['Chassis']
        if cnum not in Outfile:
            Outfile[cnum] = (args.outdir / f'{prefix}-CH{cnum:02d}-scale.txt').open('w')

        OF = Outfile[cnum]

        # write: chan, offset, slope, unit, name
        #  1, 0.0, 11834.3195266272, Pascal, CM1

        OF.write(f"{sig['Address']['Channel']}, {sig['Intercept']}, {sig['Slope']}, {sig['Egu']}, {sig['Name']}\n")

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(getargs().parse_args())
