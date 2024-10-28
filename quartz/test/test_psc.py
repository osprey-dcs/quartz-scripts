import unittest
from pathlib import Path

from .. import psc, open as qopen

_datadir = Path(__file__).parent

class TestDat(unittest.TestCase):
    def test_dat(self):
        with open(_datadir / 'mic-test-CH17-20240514-154925.dat', 'rb') as F:
            D = psc.read_dat(F)
            self.assertTupleEqual(D.shape, (9584,))
            self.assertTupleEqual(D['samp'].shape, (9584, 14, 32, 3))
            c0 = psc.get_chan(D, 0)
            c1 = psc.get_chan(D, 1)

        self.assertAlmostEqual(c0.max(), 1464.0)
        self.assertAlmostEqual(c0.min(), -2179.0)

class TestQuartz(unittest.TestCase):
    def setUp(self):
        self.q = qopen(_datadir / 'mic-test.json')

    def test_iter(self):
        infos = list(self.q)
        self.assertEqual(infos[0]['id1'], '513-BS01-DV01-CM1')
        self.assertEqual(infos[1]['id1'], '514-BS02-DV02-CM2')
        self.assertEqual(len(infos), 32)

    def test_set(self):
        D = self.q['*CM1']
        self.assertEqual(D.id1, '513-BS01-DV01-CM1')
        self.assertTupleEqual(D.shape, (134176,))
        self.assertTupleEqual(D.time.shape, (134176,))
        self.assertAlmostEqual(float(D.max()), 34.4943962097168)

        D = self.q[0]
        self.assertEqual(D.id1, '513-BS01-DV01-CM1')
