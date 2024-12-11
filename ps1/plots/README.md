# Problem set 1

### Data analysis

For each pole pair, here are the eigenvalues of the correlation matrix:

![(0.9, 0.8)](./poles_0.9_0.8/eigvals.png "(0.9, 0.8)")

![(0.95, 0.8)](./poles_0.95_0.8/eigvals.png "(0.95, 0.8)")

![(0.95, -0.9)](./poles_0.95_-0.9/eigvals.png "(0.95, -0.9)")

Here are the PSDs:

![(0.9, 0.8)](./poles_0.9_0.8/psd.png "(0.9, 0.8)")

![(0.95, 0.8)](./poles_0.95_0.8/psd.png "(0.95, 0.8)")

![(0.95, -0.9)](./poles_0.95_-0.9/psd.png "(0.95, -0.9)")

With increasingly large submatricies of the correlation matrix, the minimum and maximum eigenvalues expand from the autocorrelation to get closer to the minimum and maximum PSD.

When using $K = N_0 - L$, $\frac1KXX^T$ approximates the correlation matrix, with $N_0 - L$ being the number of points averaged for each point in the matrix. The matrix is not exactly Toeplitz because the exact vectors multiplied for corresponding correlations start at different points even if they have the same time delay. The plots of the empirical and approximate correlations are shown below:

![(0.9, 0.8)](./poles_0.9_0.8/correlation.png "(0.9, 0.8)")

![(0.95, 0.8)](./poles_0.95_0.8/correlation.png "(0.95, 0.8)")

![(0.95, -0.9)](./poles_0.95_-0.9/correlation.png "(0.95, -0.9)")

The singular values are the square roots of the coorelation values without the scaling factor $K$, so $\alpha = \frac1{\sqrt{K}} = \frac1{\sqrt{N_0 - L}}$.

The integral of the PSD is seen to be very close to the estimated value of autocorrelation.
