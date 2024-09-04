import numpy as np
from math import cos, sin, pi, sqrt, log

class BasisFunctions:
    def phi(self, x):
        return 1 if 0 <= x < 1 else 0

    def psi(self, x):
        if 0 <= x < 0.5:
            return 1
        elif 0.5 <= x < 1:
            return -1
        return 0

    def h(self, i, N):
        if i == 0:
            return self.phi

        n, k = [(n, k) for n in range(int(log(N, 2))) for k in range(2 ** n)][i - 1]
        return lambda x: 2 ** (n / 2.0) * self.psi(2 ** n * x - k)

    def v(self, h, N):
        return [h(i / float(N)) for i in range(N)]

    def Haar1_qt(self, rows, cols):
        return np.array([self.v(self.h(i, cols), rows) for i in range(cols)]).T

    def DCT_II_f(self, k, N):
        return lambda x: np.cos(pi * (x + 0.5) * k / N)

    def w(self, k, N):
        c = sqrt(2) ** np.sign(k)
        return [c * self.DCT_II_f(k, N)(i / float(N)) for i in range(N)]

    def DCT1_qt(self, rows, cols):
        return np.array([self.w(k, rows) for k in range(cols)]).T

    def DCT1_Haar1_qt(self, rows, cols):
        dct_matrix = self.DCT1_qt(rows, cols // 2)
        haar_matrix = self.Haar1_qt(rows, cols // 2)
        return np.concatenate((dct_matrix, haar_matrix), axis=1)

    def W(self, k1, k2, n, N):
        def ro(t):
            return [[cos(t), -sin(t)], [sin(t), cos(t)]]

        def theta(n, N):
            return pi * n / (2.0 * N)

        def g(k1, k2, N1, N2, v):
            return self.DCT_II_f(k1, N1)(v[0]) * self.DCT_II_f(k2, N2)(v[1])

        def W_elem(i, j):
            return g(k1, k2, 8, 8, np.dot(ro(theta(n, N)), [[i / 8.0], [j / 8.0]]))

        return [[W_elem(i, j) for j in range(8)] for i in range(8)]