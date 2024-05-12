#!/usr/bin/env python3

import struct
import time
import logging
from pathlib import Path
import numpy

_log = logging.getLogger(__name__)

_psc_body = struct.Struct('>HHIII')

_body1 = numpy.dtype([
    ('sts', '>u4'),
    ('chmask', '>u4'),
    ('seq', '>u8'),
    ('sec', '>u4'),
    ('ns', '>u4'),
    #('samp', 'u1', (samp_per_packet*3,)), # packed I24
])

_body2 = numpy.dtype([
    ('sts', '>u4'),
    ('chmask', '>u4'),
    ('seq', '>u8'),
    ('sec', '>u4'),
    ('ns', '>u4'),
    ('hihi', '>u4'),
    ('hi', '>u4'),
    ('lo', '>u4'),
    ('lolo', '>u4'),
    #('samp', 'u1', (samp_per_packet*3,)), # packed I24
])
_body = {
    0x4e41: _body1,
    0x4e42: _body2,
}

def nbits(V:int) -> int:
    R=0
    while V:
        if V&1:
            R+=1
        V>>=1
    return R

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('dats', type=Path,
                   nargs='+', default=[],
                   help='.dat file')
    return P

def show(dat: Path):
    print(dat)
    F = dat.open('rb')
    ps, msgid, msglen, rsec, rns = _psc_body.unpack(F.read(_psc_body.size))
    assert ps==0x5053, ps
    assert msgid in _body, msgid # TODO: skip unknown
    _T = _body[msgid]

    T0r = rsec*1000000000 + rns

    body = F.read(msglen)
    assert len(body)==msglen, msglen

    H = numpy.frombuffer(body[:_T.itemsize], dtype=_T)
    T0 = H["sec"][0]*1000000000 + H["ns"][0]
    nchan = nbits(H['chmask'][0])
    nsamp = (len(body) - _T.itemsize)/3
    samp_per_chan = nsamp/nchan

    ps, msgid2, msglen, rsec2, rns2 = _psc_body.unpack(F.read(_psc_body.size))
    assert ps==0x5053, ps
    assert msgid==msgid2 # TODO: support mixed

    body = F.read(msglen)
    assert len(body)==msglen, msglen

    H1 = numpy.frombuffer(body[:_T.itemsize], dtype=_T)
    T1 = H1["sec"][0]*1000000000 + H1["ns"][0]
    # time between packets / samples per packet
    dT = (T1-T0) / samp_per_chan # ns / sample
    Fsamp = 1e9 / dT

    print(f'  msgid: 0x{msgid:04x}')

    print(f'  T0: {time.ctime(H["sec"][0])} (0x{H["sec"][0]:08x}, 0x{H["ns"][0]:08x})')
    print(f'  Tr: {time.ctime(rsec)} Tr-T0: {(T0r-T0)/1e9} s')

    print(f'  #chan: {nchan} #samp/pkt: {nsamp} samp/chan: {samp_per_chan}')

    print(f'  Fsamp: {Fsamp} Hz')

def main(args):
    for dat in args.dats:
        show(dat)

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(getargs().parse_args())
