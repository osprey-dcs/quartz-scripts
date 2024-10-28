import unittest

import numpy

from .. import DataChannel

class TestChan(unittest.TestCase):
    def setUp(self):
        t = self.t = numpy.linspace(1, 1+numpy.pi*10, num=256)
        x = self.x = numpy.sin(t).view(DataChannel)
        assert x.shape==(256,)
        x._info = {
            'abscissa_min': t[0],
            'abscissa_inc': t[1]-t[0],
        }

    def test_time(self):
        t = self.x.time
        assert t.shape==self.x.shape
        assert (t-self.t).max() < self.x.abscissa_inc/1024

    def test_slice(self):
        x = self.x.slice(2, 2+numpy.pi*5)
        self.assertAlmostEqual(x.abscissa_min, 2.1087974071493396)
        assert x.abscissa_inc==self.t[1]-self.t[0]
        assert x.shape==(127,)

    def test_decimate(self):
        x = self.x.decimate(2)
        assert x.abscissa_inc==(self.t[1]-self.t[0])*2
        assert x.abscissa_min==self.t[0]
        assert x.shape==(128,)
