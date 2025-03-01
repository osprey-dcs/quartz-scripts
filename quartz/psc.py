"""Reader/Writer for PSCDRV "fast" data files

http://mdavidsaver.github.io/pscdrv/udpfast.html#file-format

Specialized for Quartz digitizer msgid
"""

import struct
from collections import namedtuple

import numpy

from . import DataSet, DataChannel

_psc_hdr = struct.Struct('>2sHI')

def _msg_layout(msgid: int, bodylen: int) -> numpy.dtype:
    """Quartz data packet format, including PSC UDP header
    """
    _T = [
        ('ps', '>u2'),
        ('msgid', '>u2'),
        ('blen', '>u4'),
        ('rsec', '>u4'),
        ('rns', '>u4'),
        # body length includes the following
        ('sts', '>u4'),
        ('chmask', '>u4'),
        ('seq', '>u8'),
        ('sec', '>u4'),
        ('ns', '>u4'),
    ]
    if msgid==0x4e41:
        pass
    elif msgid==0x4e42:
        _T += [
            ('hihi', '>u4'),
            ('hi', '>u4'),
            ('lo', '>u4'),
            ('lolo', '>u4'),
        ]
    else:
        raise ValueError(f'Unsupported msgid 0x{msgid:04x}')

    samp_len = bodylen - (numpy.dtype(_T).itemsize - 16)
    assert samp_len >= 3, samp_len
    nsamp_per_chan, rem = divmod(samp_len, 3*32) # TODO: assumes chmask==0xffffffff
    assert rem == 0, (msgid, bodylen, numpy.dtype(_T).itemsize, nsamp_per_chan, rem)
    _T += [
        ('samp', 'u1', (nsamp_per_chan, 32, 3)), # packed I24, channels interleaved for each time point
    ]
    return numpy.dtype(_T)

def read_dat(file):
    """Read packet stream from .dat file w/o decoding samples array
    """
    pos = file.tell()
    ps, msgid, blen = _psc_hdr.unpack(file.read(8))
    assert ps == b'PS', ps
    file.seek(pos)

    T = _msg_layout(msgid, blen)

    F = numpy.fromfile(file, dtype=T)

    # TODO: assumes all with identical msgid (true so far...)
    # the following checks effectively force the entire file into RAM
    assert numpy.all(F['ps']==0x5053)
    assert numpy.all(F['msgid']==msgid)
    assert numpy.all(F['blen']==blen)
    # TODO: assumes all channels included.  (FW supports any non-zero mask, so far IOC always selects all)
    assert numpy.all(F['chmask']==0xffffffff)

    dSEQ = numpy.diff(F['seq'])
    if numpy.any(dSEQ!=1):
        raise RuntimeError(f'{file.name} missing packets: {numpy.where(dSEQ)}')

    return F

def get_chan(F: numpy.ndarray, chan: int) -> numpy.ndarray:
    """Extract a single channel from the provided message stream.

    :param F: Input msg stream
    :param chan: Channel index 0->31
    """
    assert F[0]['chmask'] & (1<<chan), (F[0]['chmask'], chan)
    assert F[0]['chmask']==0xffffffff

    S24 = F['samp'][:,:,chan,:] # (npkt, nsamp_per_chan, 3)
    S32 = numpy.ndarray(S24.shape[:2] + (4,), dtype='u1')
    S32[...,1:] = S24
    S32[...,0] = numpy.bitwise_and(S24[...,0], 0x80)/128*255 # sign extend
    del S24

    S32 = S32.view('>i4') # (npkt, nsamp_per_chan, 1)

    return S32.astype('f4').flatten()

SetInfo = namedtuple("SetInfo", ['idx', 'info'])

class QuartzRaw(DataSet):
    def __init__(self, file):
        try:
            self.__data = D = read_dat(file)
        finally:
            file.close()

        assert D['samp'].shape[2:]==(32, 3)
        samp_per_pkt = D['samp'].shape[1]
        # times of first sample in each packet
        T = D['sec'] + D['ns']*1e-9
        dT = numpy.diff(T).mean() / samp_per_pkt
        # TODO: account for extra samp_per_pkt-1

        self._index = []
        for n in range(1, 33): # 1's index
            self._index.append(SetInfo(
                idx=n,
                info={
                    'abscissa_min': 0.0,
                    'abscissa_inc': dT,
                },
            ))

    def _read_set(self, idx:int):
        _idx, info = self._index[idx]
        chan = get_chan(self.__data, idx)

        chan = chan.view(DataChannel)
        chan._info = info
        return chan
