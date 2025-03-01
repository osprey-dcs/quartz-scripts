
import io
from fnmatch import fnmatch

import numpy
import scipy.signal as sig

class DataChannel(numpy.ndarray):
    """Data set which is augmented with UFF meta-data, accessible as attributes
    """
    _info = _abscissa = None
    def __getattr__(self, k):
        try:
            return self._info[k]
        except:
            raise AttributeError(k)

    @property
    def abscissa(self) -> numpy.ndarray:
        '''Return abscissa (usually timebase) array'''
        R = self._abscissa
        if R is None:
            assert self.ndim==1
            start, step = self._info['abscissa_min'], self._info['abscissa_inc']
            self._abscissa = R = numpy.arange(self.shape[0], dtype='f8')*step + start
        return R

    # aka. in the most common case...
    time = abscissa

    def slice(self, start=None, end=None) -> 'DataChannel':
        '''Return DataChannel sliced along abscissa
        '''
        absc = self.abscissa
        mask = numpy.logical_and(
            absc >= start,
            absc < end,
        )
        absc = absc[mask]

        info = self._info.copy()
        info['abscissa_min'] = absc[0]

        R = self[mask].view(self.__class__)
        R._info = info
        R._abscissa = absc
        return R

    def decimate(self, n, **kws) -> 'DataChannel':
        '''Apply scipy.signal.decimate
        '''
        R = sig.decimate(self, n, **kws).view(self.__class__)
        R._info = self._info.copy()
        R._info['abscissa_inc'] *= n
        R._abscissa = None
        return R

    # TODO: add __round__()

class DataSet:
    """Interface to access a set of channels

    Open...

    >>> from quartz import open as qopen
    >>> U = qopen('some.uff')

    Extract a single set where any ID line matches pattern "Mic*"

    >>> S = U['Mic*']

    Time plot

    >>> plt.plot(S.time, S)
    """
    # impl. notes
    # self._index is a list of namedtuple with at least attribute .info dict
    _index: list = None

    def close(self):
        pass

    def __enter__(self):
        return self
    def __exit__(self,A,B,C):
        self.close()

    def __iter__(self):
        """Yield dict of channel meta-data.
        """
        return iter([S.info for S in self._index])

    def info(self, key) -> dict:
        """Return dict of channel meta-data for the single matching channel
        """
        return self._index[self._lookup_set(key)].info

    def infos(self, key) -> [dict]:
        """Return list of dict of channel meta-data for any matching channel
        """
        return [self._index[idx].info for idx in self._lookup_set(key, first=False)]

    def __getitem__(self, key) -> DataChannel:
        """Load single matching dataset
        """
        return self._read_set(self._lookup_set(key))

    def sets(self, key) -> [DataChannel]:
        """Load all matching datasets
        """
        return [self._read_set(idx) for idx in self._lookup_set(key, first=False)]

    def _lookup_set(self, key, first=True) -> int:
        """Lookup index from matching ID* line
        """
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            R = []
            for idx,S in enumerate(self._index):
                for K,V in S.info.items():
                    if K.startswith('id') and fnmatch(V, key):
                        R.append(idx)
            if first:
                if len(R)==0:
                    raise ValueError(f'No such dataset {key}')
                elif len(R)!=1:
                    raise ValueError(f'More than one dataset matches: {key}: {R[:3]}')
                return R[0]
            else:
                return R
        else:
            raise TypeError("lookup by index or ID line")

    def _read_set(self, idx:int):
        raise NotImplementedError()

def open(fname: str) -> DataSet:
    """Read in a data set from UFF or Quartz set HDR file
    """
    F = io.open(str(fname), 'rb')
    try:
        magic = F.read(2)
        F.seek(0)
        if magic==b'PS':
            from .psc import QuartzRaw
            return QuartzRaw(F)

        magic = F.readline().rstrip()
        F.seek(0)
        if magic[:1]==b'{': # JSON HDR file
            from .quartz import Quartz
            return Quartz(F)
        elif magic==b'    -1':
            from .uff import UFF
            return UFF(F)
        else:
            raise RuntimeError(f'{fname!r} has bad magic {magic!r}')
    except:
        F.close()
        raise

def getargs():
    from argparse import ArgumentParser
    P = ArgumentParser()
    P.add_argument('file', help='UFF or .hdr file')
    return P

def main():
    from pprint import pprint
    args = getargs().parse_args()
    with open(args.file) as F:
        for S in F:
            pprint(S)
