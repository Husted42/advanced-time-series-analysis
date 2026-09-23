'''
    This file contains the helper function that the proffesor used in his R-script
'''
import numpy as np
import matplotlib.pyplot as plt


def local_polynomial_predict(x_train, y_train, x0, bandwidth, degree=2):
    """
    Local polynomial regression evaluated at x0.

    Similar idea to R's loess():
    observations close to x0 receive larger weights.

    Parameters
    ----------
    x_train : array-like
        Predictor values.
    y_train : array-like
        Response values.
    x0 : float
        Point where the regression should be evaluated.
    bandwidth : float
        Width of the local neighbourhood.
    degree : int
        Polynomial degree. degree=2 corresponds to local quadratic regression.

    Returns
    -------
    float
        Estimated y-value at x0.
    """

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    # Distance from the point we want to estimate
    distance = np.abs(x_train - x0)

    # Only observations within the bandwidth contribute
    u = distance / bandwidth

    # Tricube kernel, commonly used by LOESS
    weights = np.where(
        u < 1,
        (1 - u**3)**3,
        0
    )

    # If no observations are inside the window
    if np.sum(weights) == 0:
        return np.mean(y_train)

    # Centre x around x0.
    #
    # For degree=2:
    # y = beta_0 + beta_1(x-x0) + beta_2(x-x0)^2
    #
    # At x=x0:
    # y_hat = beta_0
    x_centered = x_train - x0

    X = np.column_stack([
        x_centered**d
        for d in range(degree + 1)
    ])

    # Weighted least squares
    W = np.diag(weights)

    beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y_train

    return beta[0]

def leave_one_out(x, y, bandwidths=None, plot_fits=False):
    """
    Leave-one-out cross validation for selecting the bandwidth
    of a local quadratic regression.

    Parameters
    ----------
    x : array-like
        Predictor variable.
    y : array-like
        Response variable.
    bandwidths : array-like, optional
        Bandwidths to test.
    plot_fits : bool
        Plot RSS as a function of bandwidth.

    Returns
    -------
    float
        RSS obtained using the best bandwidth.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if bandwidths is None:
        data_range = np.max(x) - np.min(x)

        bandwidths = np.linspace(
            0.05 * data_range,
            1.0 * data_range,
            30
        )

    rss_values = []

    for bandwidth in bandwidths:

        predictions = np.zeros(len(x))

        for i in range(len(x)):

            # Leave observation i out
            mask = np.arange(len(x)) != i

            x_train = x[mask]
            y_train = y[mask]

            predictions[i] = local_polynomial_predict(
                x_train,
                y_train,
                x[i],
                bandwidth,
                degree=2
            )

        # Leave-one-out RSS
        rss = np.sum((y - predictions)**2)

        rss_values.append(rss)

    rss_values = np.asarray(rss_values)

    # Bandwidth giving smallest RSS
    best_index = np.argmin(rss_values)
    best_bandwidth = bandwidths[best_index]
    best_rss = rss_values[best_index]

    if plot_fits:
        plt.figure(figsize=(8, 5))

        plt.plot(
            bandwidths,
            rss_values,
            marker="o"
        )

        plt.axvline(
            best_bandwidth,
            linestyle="--"
        )

        plt.xlabel("Bandwidth")
        plt.ylabel("Leave-one-out RSS")
        plt.title("Bandwidth selection")

        plt.show()

    return best_rss

def calculate_ldf(
    x,
    lags,
    n_boot=30,
    plot_fits=False,
    random_state=None
):
    """
    Estimate the LDF and a bootstrap confidence threshold.
    """

    x = np.asarray(x)
    lags = np.asarray(lags)
    rng = np.random.default_rng(random_state)

    ldf_values = np.zeros(len(lags))

    # Estimate the LDF for each lag.
    for i, k in enumerate(lags):

        print(
            f"Calculating LDF no. {i + 1} of {len(lags)}"
        )

        current_x = x[k:]
        lagged_x = x[:-k]

        rss_k = leave_one_out(
            x=lagged_x,
            y=current_x,
            plot_fits=plot_fits
        )

        rss = np.sum(
            (current_x - np.mean(current_x))**2
        )

        ldf_values[i] = (rss - rss_k) / rss

    # Bootstrap lag 1 data to estimate the null distribution.
    iid_values = np.zeros(n_boot)

    for i in range(n_boot):

        print(
            f"Calculating bootstrap no. {i + 1} of {n_boot}"
        )

        sample_size = min(len(x), 100)
        xr = rng.choice(
            x,
            size=sample_size,
            replace=True
        )

        current_x = xr[1:]
        lagged_x = xr[:-1]

        rss_k = leave_one_out(
            x=lagged_x,
            y=current_x
        )

        rss = np.sum(
            (current_x - np.mean(current_x))**2
        )

        iid_values[i] = (rss - rss_k) / rss

    confidence_level = np.quantile(
        iid_values,
        0.95
    )

    return ldf_values, confidence_level

def plot_ldf(lags, ldf_values, confidence_level, filename = None):
    lags = np.asarray(lags)
    ldf_values = np.asarray(ldf_values)

    # Add lag 0 as a reference point.
    plot_lags = np.concatenate([[0], lags])
    plot_values = np.concatenate([[1], ldf_values])

    plt.figure(figsize=(10, 5))

    plt.vlines(
        plot_lags,
        0,
        plot_values
    )

    plt.scatter(
        plot_lags,
        plot_values
    )

    plt.axhline(
        0,
        linestyle="-",
        linewidth=1
    )

    plt.axhline(
        confidence_level,
        linestyle="--",
        label="95% bootstrap level"
    )

    plt.xticks(plot_lags)
    plt.ylim(-1, 1)

    plt.xlabel("Lag")
    plt.ylabel("LDF")
    plt.title("Lag Dependence Functions")

    plt.legend()

    if filename:
        plt.savefig(f"assets/{filename}")
    plt.show()