import affprop as ap
import numpy as np
import unittest


class MainTest(unittest.TestCase):

    def setUp(self):
        self.M = np.array([[1, 2], [3, 4], [5, 6], [1, 1], [3, 3]])
        self.M_small = np.array([[1, 2], [3, 4], [5, 6]])
        self.S_euclid = np.array([
            [0, -8, -32, -1, -5],
            [0, 0, -8, -13, -1],
            [0, 0, 0, -41, -13],
            [0, 0, 0, 0, -8],
            [0, 0, 0, 0, 0]
        ])
        self.S_cos = np.array([
            [1, 0.9838699, 0.9734172, 0.9486833, 0.9486833],
            [0, 1, 0.9986877, 0.9899495, 0.9899495],
            [0, 0, 1, 0.9958932, 0.9958932],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1]
        ])
        self.S_euclid = self.S_euclid + self.S_euclid.T
        self.S_cos = self.S_cos + self.S_cos.T - np.eye(self.S_cos.shape[0])
        self.S_median = np.median(self.S_euclid)
        self.S_min = np.min(self.S_euclid)

    def test_similarity(self):
        S1 = ap.compute_similarity(self.M, measure="eucl")
        S2 = ap.compute_similarity(self.M, measure="cos")
        np.testing.assert_array_equal(S1, self.S_euclid)
        np.testing.assert_array_almost_equal(S2, self.S_cos)

    def test_give_preferences(self):
        S = ap.compute_similarity(self.M, measure="eucl")
        vec = np.array([1, 2, 3, 4, 5])
        scalar = 42
        S_euclid = self.S_euclid.copy()

        S = ap.give_preferences(S, preference="median")
        np.fill_diagonal(S_euclid, self.S_median)
        np.testing.assert_array_equal(
            S,
            S_euclid
        )

        S = ap.give_preferences(S, preference="min")
        np.fill_diagonal(S_euclid, self.S_min)
        np.testing.assert_array_equal(
            S,
            S_euclid
        )

        S = ap.give_preferences(S, preference=scalar)
        np.fill_diagonal(S_euclid, scalar)
        np.testing.assert_array_equal(
            S,
            S_euclid
        )

        S = ap.give_preferences(S, preference=vec)
        np.fill_diagonal(S_euclid, vec)
        np.testing.assert_array_equal(
            S,
            S_euclid
        )

    @unittest.skip("needs fixing")
    def test_responsibility(self):
        S = ap.compute_similarity(self.M_small, measure="eucl")
        S = ap.give_preferences(S, preference=1)
        R0 = np.zeros(S.shape)
        A0 = np.zeros(S.shape)
        R = ap.compute_responsibility(S, R0, A0, dampfac=0)
        R_test = np.array([[]])

    def test_availability(self):
        S = ap.compute_similarity(self.M_small, measure="eucl")
        S = ap.give_preferences(S, preference=1)
        R = np.array([[9, -9, -33], [-9, 9, -9], [-33, -9, 9]])
        A0 = np.zeros(S.shape)
        A = ap.compute_availability(R, A0, dampfac=0)
        A_test = np.zeros(S.shape)
        np.testing.assert_array_equal(A, A_test)


if __name__ == "__main__":
    unittest.main()
