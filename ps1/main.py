from math import ceil, log

from scipy import signal
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt


N_iter = 50
N_0 = 10000
L = 20

# "I prescribed three pole pairs above. Repeat EVERYTHING for each pole set."
for pole_pair in [(0.9, 0.8), (0.85, 0.8), (0.95, -0.9)]:
    transient = 0.01
    max_abs_pole = max([abs(pole) for pole in pole_pair])
    N_init = ceil(log(transient, max_abs_pole))

    v = np.random.normal(0, 1, N_init + N_0)
    b, a = signal.zpk2tf([], [pole for pole in pole_pair], 1)
    x = signal.lfilter(b, a, v)
    x = x[N_init:]

    for M in [2, 4, 10]:
        X = la.toeplitz(x[M::-1], x[M:])

        x_vec_L = la.toeplitz(x[L::-1], x[L:2*L+1])
        R_L_plus_1 = x_vec_L.T @ x_vec_L

        r = R_L_plus_1[0]
        omega = np.linspace(-np.pi, np.pi, 1000)
        S = r[0] * np.ones_like(omega)
        for m in range(1, L+1):
            S += 2 * r[m] * np.cos(m*omega)
        # S = np.abs(np.fft.fft(np.concatenate([r[:0:-1], r])))
        breakpoint()
        assert S.imag.max() < 1e-12
        S = S.real
        plt.plot(S)
        plt.show()

        for m in range(1, L+2):
            R_m = R_L_plus_1[:m+1, :m+1]
            eigvals = la.eigvals(R_m)

        eigvals = la.eigvals(R_L_plus_1)


signal.lfilter()