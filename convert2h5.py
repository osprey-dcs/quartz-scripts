#!/usr/bin/env python3

import sys
import os
import time
import struct
import logging
from pathlib import Path

import numpy
import h5py

_log = logging.getLogger(__name__)

_head = struct.Struct('>HHIII')

def cal_default():
    R = numpy.zeros((2, 32), dtype='f8')
    R[1,:] = 1.0
    return R

def cal_file(fname: str): # -> [2, 32], [units], [label]
    R = cal_default()
    Es = [None]*32
    Ls = [None]*32
    with open(fname, 'r') as F:
        for i, L in enumerate(F):
            try:
                L = L.strip()
                if len(L)==0 or L[0]=='#':
                    continue
                cols = [e.strip() for e in L.split(',')]
                idx, off, slo = cols[:3]
                idx = int(idx)-1
                if len(cols)>3:
                    Es[idx] = cols[3]
                if len(cols)>4:
                    Ls[idx] = cols[4]
                R[:, idx] = (off, slo)
            except:
                _log.exception(f"{fname} on line {i+1}")
                raise
    return R, Es, Ls

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('--calib', type=cal_file, default=cal_default())
    P.add_argument('--scale', type=cal_file, default=cal_default())
    P.add_argument('--freq', type=float, default=250e3,
                   help='Sampling frequency')
    P.add_argument('--title')
    P.add_argument('-o', '--output')
    P.add_argument('dat', nargs='+',
                   help='.dat files')
    P.add_argument('--block-size', type=int, default=64*1024*1024)
    return P

def peek_sizes(F):
    ps, msgid, blen, rsec, rns = _head.unpack(F.read(_head.size))
    assert ps==0x5053, ps
    assert msgid==0x4e41, msgid
    pktlen = _head.size + blen
    assert blen>=4*6, blen
    blen -= 4*6
    assert blen%3==0, blen
    return pktlen, blen//3

def combine_scales(args): # -> [2, 32], Es, Ls
    Es = [None]*32
    Ls = [None]*32

    A, uA, lA = args.calib
    E, uE, lE = args.scale
    # V = ASLO*ADC + AOFF
    # EU= ESLO*V   + EOFF
    #
    # EU= ESLO*(ASLO*ADC + AOFF) + EOFF
    # EU= (ESLO*ASLO)*ADC + (ESLO*AOFF + EOFF)
    AOFF, ASLO = A[0,:], A[1,:]
    EOFF, ESLO = E[0,:], E[1,:]
    scal = numpy.ndarray(E.shape, dtype=E.dtype)
    scal[1,:] = ESLO*ASLO
    scal[0,:] = ESLO*AOFF + EOFF

    for i in range(32):
        Ls[i] = lE[i] or lA[i] or f'ch{i+1}'
        Es[i] = uE[i] or uA[i] or 'adc'

    return scal, Es, Ls

def main(args):
    scale, units, labels = combine_scales(args) # [2, 32]

    # peek at first header in first file assuming
    with open(args.dat[0], 'rb') as IN:
        pktlen, samp_per_packet = peek_sizes(IN)
        # pktlen input bytes should yield 4*samp_per_packet output bytes

    _log.debug('pktlen = %s, samp_per_packet = %s', pktlen, samp_per_packet)

    pkt_per_block = args.block_size // pktlen
    samp_per_block = samp_per_packet * pkt_per_block
    _log.debug('pkt_per_block = %s, samp_per_block = %s', pkt_per_block, samp_per_block)

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
        ('samp', 'u1', (samp_per_packet*3,)), # packed I24
    ])
    _log.debug('sizeof(T) = %s', _T.__sizeof__())
    assert _T.itemsize==pktlen, (_T.__sizeof__(), pktlen)

    args.block_size -= args.block_size%pktlen

    with h5py.File(args.output, 'w') as OUT:
        DS = OUT.create_dataset('adc',
                                dtype='f4',
                                shape=(0, 32),
                                chunks=(samp_per_block, 32),
                                maxshape=(None, 32),
        )

        DS.attrs['units'] = units
        DS.attrs['labels'] = labels
        DS.attrs['calib'] = args.calib[0]
        DS.attrs['scale'] = args.scale[0]
        DS.attrs['Fsamp'] = args.freq

        idx = 0

        nextSEQ = None
        T0 = None
        rT0 = time.monotonic()
        for ifile in args.dat:
            _log.info('Read %s', ifile)
            with open(ifile, 'rb') as IN:
                fsizeM = IN.seek(0, 2) // 1024//1024
                IN.seek(0)
                fileT0 = None
                while True:
                    rT1 = time.monotonic()
                    _log.info('. %.2f %s/%s', rT1-rT0, IN.tell()//1024//1024, fsizeM)
                    rT0 = rT1
                    blk = IN.read(args.block_size)
                    if len(blk)==0:
                        break

                    pkts = numpy.frombuffer(blk, dtype=_T)
                    assert numpy.all(pkts['ps']==0x5053)
                    assert numpy.all(pkts['msgid']==0x4e41)
                    assert numpy.all(pkts['chmask']==0xffffffff) # assume all channels enabled

                    # check sequence numbers
                    # ... for gaps between files/blocks
                    dSEQ = numpy.diff(pkts['seq'])
                    if nextSEQ is not None and nextSEQ+1!=pkts['seq'][0]:
                        _log.warning('%d -> %d missing packets between file/block',
                                     nextSEQ, pkts['seq'][0])
                    nextSEQ = pkts['seq'][-1]
                    # ... for gaps within a block
                    if not numpy.all(dSEQ==1):
                        _log.warning('%s packets missing', numpy.sum(dSEQ!=1))

                    S24 = pkts['samp']
                    S24 = S24.reshape((S24.shape[0], -1, 3)) # #samp, #pkt, 3
                    assert samp_per_packet==S24.shape[1], (samp_per_packet, S24.shape)

                    S32 = numpy.zeros(S24.shape[:2]+(4,), dtype=S24.dtype) # #samp, #pkt, 4
                    S32[...,1:] = S24
                    S32[...,0 ] = numpy.bitwise_and(S24[...,0], 0x80)/128*255 # sign extend
                    del S24

                    S32 = S32.view('>i4').flatten() # concat packets
                    S32 = S32.reshape((-1,32)) # #samp, 32

                    # apply scaling
                    F32 = S32.astype('f4')
                    del S32
                    F32 *= scale[1,:][None,:].repeat(F32.shape[0], 0)
                    F32 += scale[0,:][None,:].repeat(F32.shape[0], 0)

                    pktT0 = pkts['sec'][0] + pkts['ns'][0]*1e-9

                    if fileT0 is None:
                        # time of first sample in first packet of current file
                        fileT0 = pktT0

                    if T0 is None:
                        DS.attrs['T0'] = T0 = fileT0
                        DS.attrs['T0_desc'] = 'time of first sample in acquisition'

                    DS.resize(idx+F32.shape[0], 0)
                    DS[idx:, :] = F32
                    idx += F32.shape[0]
                    del F32

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(getargs().parse_args())
