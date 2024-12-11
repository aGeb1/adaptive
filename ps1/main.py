from math import ceil, log
import warnings

from scipy import signal
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

N_iter = 50
N_0 = 10000
L = 20

pole_pairs = [np.array(pole_pair) for pole_pair in [(0.9, 0.8), (0.85, 0.8), (0.95, -0.9)]]

for poles in pole_pairs:
    print(f"poles: {poles}")

    # Determine N_init
    transient = 0.01
    max_abs_pole = max([abs(pole) for pole in poles])
    N_init = ceil(log(transient, max_abs_pole))

    # Generate signal
    v = np.random.normal(0, 1, N_init + N_0)
    b, a = signal.zpk2tf([], poles, 1)
    x = signal.lfilter(b, a, v)
    x = x[N_init:]

    # Calculate correlation matrix
    K = np.array([poles[0]/(1 - poles[0]**2), -poles[1]/(1 - poles[1]**2)])
    K /= poles[0] * (1 + poles[1]**2) - poles[1] * (1 + poles[0]**2)
    m = np.array(range(L+1))
    r = K[0] * poles[0] ** m + K[1] * poles[1] ** m
    R = la.toeplitz(r)

    # Calculate PSD
    omega = np.linspace(-np.pi, np.pi, 1000)
    z = np.exp(1j*omega)
    S = 1/((1-poles[0]*z)*(1-poles[0]/z)*(1-poles[1]*z)*(1-poles[1]/z))
    assert S.imag.max() < 1e-11
    S = S.real

    S_min, S_max = min(S), max(S)
    print(f"\tS_min: {S_min:.4f}, S_max: {S_max:.4f}\n")

    plt.plot(omega, S)
    plt.savefig(f"plots/poles_{poles[0]}_{poles[1]}/psd.png")
    plt.close()
    print("\tPSD plotted.\n")

    # Calculate eigenvalues of coorelation matricies
    eigvals = la.eigvals(R)
    assert eigvals.imag.max() == 0
    plt.stem(eigvals)
    plt.savefig(f"plots/poles_{poles[0]}_{poles[1]}/eigvals.png")
    plt.close()
    print("\tEigenvalues plotted.\n")

    print("\tSubmatrix eigenvalues:")
    for m in range(L+1):
        eigvals = la.eigvals(R[:m+1, :m+1]).real
        lambda_min, lambda_max = min(eigvals), max(eigvals)
        assert S_min < lambda_min
        assert lambda_max < S_max
        print(f"\t\tm: {m+1}, lambda_min: {lambda_min.real:.4f}, lambda_max: {lambda_max.real:.4f}")

    print("\n")




    # for M in [2, 4, 10]:
    #     X = la.toeplitz(x[M::-1], x[M:])

#         eigvals = la.eigvals(R_L_plus_1)


# signal.lfilter()
