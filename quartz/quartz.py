
import json
import logging
from collections import namedtuple
from pathlib import Path
import struct

import numpy

from . import psc, DataSet, DataChannel

_jhdr = struct.Struct('<IIIQ')

_log = logging.getLogger(__name__)

SetInfo = namedtuple("SetInfo", ['idx', 'info'])

class Quartz(DataSet):
    def __init__(self, file):
        if not hasattr(file, 'readline'): # str or Path
            file = open(file, 'rb')

        self._base = Path(file.name).parent # directory containing HDR file
        index = self._json = json.load(file)
        Fsamp = index["SampleRate"]

        self._index = []
        for idx, sig in enumerate(index['Signals']):
            info = {
                'abscissa_egu': 's',
                'abscissa_inc': 1/Fsamp,
                'abscissa_label': 'Time',
                'abscissa_min': 0.0, # TODO: could include per-channel skew (< 100ns)
                'abscissa_spacing': 1,
                'abscissa_stype': 17,
                'abscissa_z': 0.0,
                'egu': sig['Egu'],
                #'fp': 2,
                'id1': sig['Name'],
                'id2': sig['Desc'],
                'id3': 'NONE',
                'id4': 'NONE',
                'id5': 'NONE',
                'label': sig['Desc'],
            }

            self._index.append(SetInfo(idx, info))

    def close(self):
        self._index = []
        self._fp.close()

    def _read_set(self, idx:int):
        idx, info = self._index[idx]

        sig = self._json['Signals'][idx]
        slope, offset = sig['Slope'], sig['Intercept']

        # prefer .j file when available
        jfile = sig.get('OutDataFile')
        if jfile is not None:
            # read .j channel data
            try:
                J = open(self._base / jfile, 'rb')
                jhdr = _jhdr.unpack(J.read(_jhdr.size))
                if jhdr[0]!=1:
                    raise RuntimeError('Unsupported J version {jhdr}')

                jsize = jhdr[3]
                F32 = numpy.fromfile(J, dtype='<i4', count=jsize//4).astype('f4')
                F32 *= slope
                F32 += offset
                F32 = F32.view(DataChannel)
                F32._info = info
                return F32

            except:
                _log.exception(f'unable to open {jfile!r}')
                # fall through to try .dat

        chas, chan = sig['Address']['Chassis'], sig['Address']['Channel'] # 1-indexed

        # lookup .dat file for chassis
        datfiles, = [chassis['Dat'] for chassis in self._json['Chassis'] if chassis['Chassis']==chas]

        if len(datfiles)>1:
            _log.warning('.dat file concat not implemented, ignoring additional files: %r', datfiles[1:])

        with open(self._base / datfiles[0], 'rb') as F:
            pkts = psc.read_dat(F) # TODO: cache most recently opened file

            F32 = psc.get_chan(pkts, chan-1)

        F32 *= slope
        F32 += offset
        F32 = F32.view(DataChannel)
        F32._info = info
        return F32
