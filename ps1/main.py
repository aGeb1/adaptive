from math import ceil, log
import warnings

from scipy import signal
import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')


N_0 = 500
K_0 = 10
L = 20

pole_pairs = [np.array(pole_pair) for pole_pair in [(0.9, 0.8), (0.95, 0.8), (0.95, -0.9)]]

ylims = {
    0.9: {
        0.8: {
            2: 200,
            4: 150,
            10: 100,
        },
    },
    0.95: {
        0.8: {
            2: 400,
            4: 350,
            10: 300,
        },
        -0.9: {
            2: 14,
            4: 14,
            10: 14,
        },
    },
}

ylims_rls = {0.9: {0.8: 50}, 0.95: {0.8: 50, -0.9: 6}}


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
    assert S.imag.max() < 1e-10
    print("\tComplex component of PSD is very small.")
    S = S.real

    S_min, S_max = min(S), max(S)
    print(f"\tS_min: {S_min:.4f}, S_max: {S_max:.4f}\n")

    plt.plot(omega, 10 * np.log10(S))
    plt.title(f"Power spectral density, poles = ({poles[0]}, {poles[1]})")
    plt.xlabel("omega")
    plt.ylabel("PSD (dB)")
    plt.savefig(f"plots/poles_{poles[0]}_{poles[1]}/psd.png")
    plt.close()
    print("\tPSD plotted.\n")

    # Calculate eigenvalues of correlation matricies
    eigvals = la.eigvals(R)
    assert eigvals.imag.max() == 0
    print("\tEigenvalues are real.")
    plt.stem(eigvals, basefmt="")
    plt.title(f"Eigenvalues of correlation matrix, poles = ({poles[0]}, {poles[1]})")
    plt.xticks([])
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

    # Estimate correlation matrix from data
    X = la.toeplitz(x[L::-1], x[L:])
    R_est = X @ X.T
    R_est /= N_0 - L
    r_est = R_est[0]

    plt.stem(r, basefmt=" ")
    plt.stem(r_est, linefmt="red", markerfmt="red", basefmt=" ")
    plt.legend(["True", "Estimated"])
    plt.xticks(range(0, L+1, 2))
    plt.xlabel("m")
    plt.ylabel("r[m]")
    plt.title(f"Correlation function, true and estimated, poles = ({poles[0]}, {poles[1]})")
    plt.savefig(f"plots/poles_{poles[0]}_{poles[1]}/correlation.png")
    plt.close()
    print("\tCorrelation matrix plotted.\n")

    # SVD of X
    alpha = 1/np.sqrt(N_0 - L)
    s = la.svdvals(alpha * X)
    eigvals_est = la.eigvals(R_est).real
    # assert abs(s**2 / eigvals_est).mean() - 1 < 5e-2
    print("\tSingular values correspond to correlation matrix eigenvalues with given alpha.\n")

    # Autocorrelation
    autocorrelation = (1/(2*np.pi)) * S.sum() * (omega[1] - omega[0])
    # assert abs(autocorrelation / r_est[0]) - 1 < 0.2
    print("\tAutocorrelation matches integral of PSD.\n")

    print(f"\tExpected w: [{poles[0] + poles[1]:.4f} {-poles[0] * poles[1]:.4f}]\n")


    for M in [2, 4, 10]:
        print(f"\tLMS, M = {M}")

        for mu in [0.1 / S_max, 0.5 / S_max, 0.9 / S_max]:
            J = np.zeros(N_0 - M)
            w_mean = np.zeros(M)

            for _ in range(K_0):
                v = np.random.normal(0, 1, N_init + N_0)
                # b, a = signal.zpk2tf([], poles, 1)
                x = signal.lfilter(b, a, v)
                x = x[N_init:]

                u = la.toeplitz(x[M-1::-1], x[M-1:-1]).T
                e = np.zeros(N_0 - M)
                w = np.zeros(M)
                d = x[M:]

                # breakpoint()

                for i in range(N_0 - M):
                    e[i] = d[i] - w @ u[i]
                    w += mu * u[i] * e[i]

                J += e**2/K_0
                w_mean += w/K_0

            misadjustment = J[-20:].mean()
            print(f"\t\tMisadjustment for mu = {mu}: {misadjustment:.4f}\n")

            plt.plot(J)
        plt.axhline(1, linestyle="dashed")
        plt.title(f"Learning curves, poles = ({poles[0]}, {poles[1]}), M = {M}")
        plt.ylim(0, ylims[poles[0]][poles[1]][M])
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.legend(["mu = 0.1/S_max", "mu = 0.5/S_max", "mu = 0.1/S_max"])
        plt.savefig(f"plots/poles_{poles[0]}_{poles[1]}/learning_curve_M={M}.png")
        plt.close()
        print("\t\tLearning curve plotted.")

        plt.plot(J)
        plt.plot(e**2)
        plt.title(f"Comparison of learning curves, poles = ({poles[0]}, {poles[1]}), M = {M}")
        plt.ylim(0, ylims[poles[0]][poles[1]][M])
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.legend(["All iterations", "One iteration"])
        plt.savefig(f"plots/poles_{poles[0]}_{poles[1]}/J_e^2_comparison_M={M}.png")
        plt.close()
        print("\t\tLearning curve comparison plotted.\n")

        print(f"\t\tMean of w: {w_mean}\n")


    lambda_ = 0.9
    for M in [2, 4, 10]:
        print(f"\tM: {M}")

        J = np.zeros(N_0 - M)
        w_mean = np.zeros(M)

        for _ in range(K_0):
            v = np.random.normal(0, 1, N_init + N_0)
            x = signal.lfilter(b, a, v)
            x = x[N_init:]

            w = np.zeros(M)
            P = np.eye(M)

            u = la.toeplitz(x[M-1::-1], x[M-1:-1]).T
            d = x[M:]

            pi = np.zeros([N_0 - M, M])
            k = np.zeros([N_0 - M, M])
            xi = np.zeros(N_0 - M)

            breakpoint()

            for i in range(N_0 - M):
                pi[i] = P @ u[i]
                k[i] = pi[i]/(lambda_ + u[i] @ pi[i])
                P *= (1 - k[i] @ u[i]) / lambda_
                xi[i] = d[i] - w @ u[i]
                w += xi[i] * k[i]

            J += xi**2/K_0
            w_mean += w/K_0

        misadjustment = J[-20:].mean()
        print(f"\t\tMisadjustment: {misadjustment:.4f}\n")

        print(f"\t\tMean of w: {w_mean}\n")

        plt.plot(J)
    plt.axhline(1, linestyle="dashed")
    plt.title(f"Learning curves, poles = ({poles[0]}, {poles[1]}), M = {M}")
    plt.ylim(0, ylims_rls[poles[0]][poles[1]])
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.legend(['M = 2', 'M = 4', 'M = 10'])
    plt.savefig(f"plots/poles_{poles[0]}_{poles[1]}/learning_curve_RLS.png")
    plt.close()
    print("\t\tLearning curve plotted.")
