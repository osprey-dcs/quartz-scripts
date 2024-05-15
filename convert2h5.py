#!/usr/bin/env python3

import sys
import os
import time
import struct
import logging
import json
from pathlib import Path

import numpy
import h5py

_log = logging.getLogger(__name__)

_head = struct.Struct('>HHIII')

def getargs():
    def jfile(fname):
        with open(fname, 'r') as F:
            return json.load(F)
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('-J', '--json', dest='jinfo', type=jfile, default={'Signals':[]},
                   help='JSON acquisition info file')
    P.add_argument('-C', '--chassis', type=int, default=1,
                   help='Chassis number (1 indexed)')
    P.add_argument('-o', '--output')
    P.add_argument('dat', nargs='+',
                   help='.dat files')
    P.add_argument('--block-size', type=int, default=64*1024*1024)
    return P

def peek_sizes(F):
    "Peek at first message in a file and extract sizes"
    ps, msgid, blen, rsec, rns = _head.unpack(F.read(_head.size))
    assert ps==0x5053, ps
    pktlen = _head.size + blen
    if msgid==0x4e41:
        assert blen>=4*6, blen
        blen -= 4*6
    elif msgid==0x4e42:
        assert blen>=4*10, blen
        blen -= 4*10
    else:
        raise RuntimeError(f'Unsupported msgid 0x{msgid:04x}')
    assert blen%3==0, blen
    return msgid, pktlen, blen//3

def main(args):
    Fsamp = args.jinfo['SampleRate']
    assert Fsamp>=1000 and Fsamp<=250000, Fsamp

    signals = [S for S in args.jinfo['Signals'] if S['Address']['Chassis']==args.chassis]
    assert len(signals)
    _log.info('Chassis %d has %d signals inuse', args.chassis, len(signals))
    signals.sort(key=lambda S:S['Address']['Channel']) # sort in increasing order

    def key2arr(k, defval):
        R = [defval]*32
        for S in signals:
            R[S['Address']['Channel']-1] = S.get(k, defval)
        return R

    attrs = {}
    attrs['scale'] = scale = numpy.ndarray((2,32), dtype='f4')
    scale[0, :] = key2arr('Intercept', 0.0)
    scale[1, :] = key2arr('Slope'    , 1.0)
    for K in ['Egu', 'Name', 'Desc']:
        attrs[K.lower()] = key2arr(K, '')
    for K in ['ResponseNode', 'ResponseDirection', 'Type']:
        attrs[K.lower()] = key2arr(K, 0)

    # peek at first header in first file assuming
    with open(args.dat[0], 'rb') as IN:
        msgid, pktlen, samp_per_packet = peek_sizes(IN)
        # pktlen input bytes should yield 4*samp_per_packet output bytes

    _log.debug('pktlen = %s, samp_per_packet = %s', pktlen, samp_per_packet)

    pkt_per_block = args.block_size // pktlen
    samp_per_block = samp_per_packet * pkt_per_block
    _log.debug('pkt_per_block = %s, samp_per_block = %s', pkt_per_block, samp_per_block)

    # message layout
    _T = [
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
    ]
    if msgid==0x4e42:
        _T += [
            ('hihi', '>u4'),
            ('hi', '>u4'),
            ('lo', '>u4'),
            ('lolo', '>u4'),
        ]
    _T += [
        ('samp', 'u1', (samp_per_packet*3,)), # packed I24
    ]
    _T = numpy.dtype(_T)
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

        for K,V in attrs.items():
            try:
                DS.attrs[K] = V
            except:
                _log.exception('%r = %r', K, V)
                raise

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
                    assert numpy.all(pkts['msgid']==msgid)
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
