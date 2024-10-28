"""UFF reader specialized for 58b

https://www.ceas3.uc.edu/sdrluff/
"""

import io
import logging
from collections import namedtuple

import numpy

from . import DataSet, DataChannel

__all__ = ('UFF',)

_log = logging.getLogger(__name__)

_endian = {
    1: '<', # LSBF
    2: '>', # MSBF
}

_btype = {
    2: numpy.dtype('f4'),
    4: numpy.dtype('f8'),
    # 5 complex single
    # 6 complex double
}

def _line_decoder(specmap:list, length=80):
    pos = 0
    actions = []
    fields = []
    for width, conv, name in specmap:
        if conv is not None:
            fields.append(name)
            actions.append((slice(pos, pos+width), conv, name))
        pos += width

    assert pos==length, (pos, length)

    def action(line:str) -> dict:
        return {name:conv(line[S]) for S,conv,name in actions}
    action.__qualname__ = action.__name__ = f'_decode_{name}'
    return action

_decode_58line0 = _line_decoder(
    # Format (I6,1A1,I6,I6,I12,I12,I6,I6,I12,I12)
    #    58b     1     2          11    36306944     0     0           0           0
    [
        (6 , None,  None),
        (1 , None,  None),
        (6 , lambda b:_endian[int(b)],   "endian"),
        (6 , int,   "fp"),
        (12, int,   "nlines"),
        (12, int,   "nbytes"),
        (6 , None,  None),
        (6 , None,  None),
        (12, None,  None),
        (12, None,  None),
    ],
    length=79,
)

_decode_58line6 = _line_decoder(
    # Format(2(I5,I10),2(1X,10A1,I10,I4))
    #    1         0    0         0 Volt             769   2 NONE             769   2
    [
        (5, int, 'functype'),
        (10, int, 'funcnum'),
        (5, int, 'uffvers'), # ??
        (10, int, 'loadcase'),
        (1, None, None),
        (10, lambda b:b.strip().decode(errors='ignore'), 'respname'),
        (10, int, 'respnode'),
        (4, int, 'respdir'),
        (1, None, None),
        (10, lambda b:b.strip().decode(errors='ignore'), 'refname'),
        (10, int, 'refnode'),
        (4, int, 'refdir'),
    ],
)

_decode_58line7 = _line_decoder(
    # Format(3I10,3E13.5)
    #         2   9076736         1 0.00000E+000 4.00000e-005 0.00000E+000
    [
        (10, lambda b:_btype[int(b)], 'dtype'),
        (10, int, 'npoints'),
        (10, int, 'abscissa_spacing'),
        (13, float, 'abscissa_min'),
        (13, float, 'abscissa_inc'),
        (13, float, 'abscissa_z'),
    ],
    length=69,
)

# lines 8-11 have the same form
_decode_58axisline = _line_decoder(
    # Format(I10,3I5,2(1X,20A1))
    #        17    0    0    0 Time                 s
    [
        (10, int, 'stype'),
        (5, None, None),
        (5, None, None),
        (5, None, None),
        (20, lambda b:b.strip().decode(errors='ignore'), 'label'),
        (20, lambda b:b.strip().decode(errors='ignore'), 'egu'),
    ],
    length=65,
)

SetInfo = namedtuple("SetInfo", ['hpos', 'bpos', 'info'])

class UFF(DataSet):
    """Access to a UFF file containing only 58b datasets.

    Open...

    >>> from quartz.uff import UFF
    >>> U = UFF('some.uff')

    Extract a single set where any ID line matches pattern "Mic*"

    >>> S = U['Mic*']

    Time plot

    >>> plt.plot(S.time, S)
    """
    def __init__(self, file):
        self._index = []
        if not hasattr(file, 'readline'): # str or Path
            file = open(file, 'rb')
        self._build_index(file)
        self._fp = file

    def close(self):
        self._index = []
        self._fp.close()

    def _read_set(self, idx:int) -> DataChannel:
        S = self._index[idx]
        if S.info['abscissa_spacing']!=1:
            raise RuntimeError('Unable to read dataset with uneven abscissa_spacing')
        self._fp.seek(S.bpos, io.SEEK_SET)
        A = numpy.fromfile(self._fp, dtype=S.info['dtype'], count=S.info['npoints']).view(DataChannel)
        A._info = S.info
        return A

    @staticmethod
    def _readline(fp:io.BufferedRandom):
        return fp.readline().rstrip(b'\n\r')

    def _build_index(self, fp:io.BufferedRandom):
        # unv.asc recommends searching for pairs of _marker lines
        # which requires inspecting every byte in this file right away.
        # Rather, we will incrementally read headers and skip bodies to build an index.
        # Faster for very large files.

        index = []

        while True:
            M = fp.readline()
            if len(M)==0:
                break # EoF
            elif M.rstrip(b'\n\r')!=b'    -1':
                raise ValueError(f'missing expected block marker before {fp.tell()}')

            hpos = fp.tell() # start of header, after marker

            line0 = self._readline(fp)
            if not line0.startswith(b'    58b'):
                raise ValueError('Unsupported type: {line0!r}')

            info: dict = _decode_58line0(line0)
            lines = [self._readline(fp) for _n in range(info['nlines'])]
            bpos = fp.tell() # start of body

            # copy in ID lines verbatim
            info.update({f'id{n+1}':l.decode(errors='ignore') for n,l in enumerate(lines[:5])})
            _log.debug('id1 %s', info['id1'])

            info.update(_decode_58line7(lines[7-1]))

            info.update({f'abscissa_{k}':v for k,v in _decode_58axisline(lines[8-1]).items()})
            info.update(_decode_58axisline(lines[9-1]))

            info['dtype'] = etype = info['dtype'].newbyteorder(info['endian'])

            # cross-check body size
            assert etype.itemsize*info['npoints']==info['nbytes'], info

            _log.debug('skip %d', info['nbytes'])
            epos = fp.seek(info['nbytes'], io.SEEK_CUR)
            assert bpos+info['nbytes']==epos

            if self._readline(fp)!=b'    -1':
                raise ValueError(f'missing expected block marker before {fp.tell()}')

            index.append(SetInfo(hpos, bpos, info))

        self._index = index

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('uff')
    return P

def main():
    args = getargs().parse_args()
    with UFF(args.uff) as U:
        for S in U._index:
            print(S.info['id1'])

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    main()
