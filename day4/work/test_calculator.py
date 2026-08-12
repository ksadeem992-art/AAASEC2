import unittest
from calculator import add, sub, mul, div

class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)

    def test_sub(self):
        self.assertEqual(sub(5, 3), 2)
        self.assertEqual(sub(0, 5), -5)
        self.assertEqual(sub(-1, -1), 0)

    def test_mul(self):
        self.assertEqual(mul(3, 4), 12)
        self.assertEqual(mul(-2, 3), -6)
        self.assertEqual(mul(0, 5), 0)

    def test_div(self):
        self.assertEqual(div(10, 2), 5)
        self.assertEqual(div(-6, 2), -3)
        self.assertEqual(div(0, 5), 0)

    def test_div_by_zero(self):
        with self.assertRaises(ValueError):
            div(1, 0)

if __name__ == '__main__':
    unittest.main()