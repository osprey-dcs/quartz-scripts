
import unittest
from pathlib import Path

from .. import open

_datadir = Path(__file__).parent

class TestUFF(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.u = open(_datadir / 'Sample_UFF58b_bin.uff')

    def tearDown(self):
        del self.u

    def test_iter(self):
        infos = list(self.u)
        self.assertEqual(infos[0]['id1'], 'Mic 01.0Scalar')
        self.assertEqual(len(infos), 1)

    def test_set(self):
        D = self.u['Mic*']
        self.assertEqual(D.id1, 'Mic 01.0Scalar')
        self.assertTupleEqual(D.shape, (79292,))
        self.assertTupleEqual(D.time.shape, (79292,))
        self.assertAlmostEqual(float(D.max()), 0.1174807)

        D = self.u[0]
        self.assertEqual(D.id1, 'Mic 01.0Scalar')

    def test_sets(self):
        S = self.u.sets('Mic*')
        self.assertEqual(S[0].id1, 'Mic 01.0Scalar')
        self.assertEqual(len(S), 1)

    def test_err(self):
        with self.assertRaises(TypeError):
            self.u[object()]
        with self.assertRaises(IndexError):
            self.u[42]
