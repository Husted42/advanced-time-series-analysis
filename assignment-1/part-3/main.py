import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(1)

omega = 0.2
alpha = 0.5
n = 10000

Z = rng.standard_normal(n)
X = np.zeros(n)

for t in range(1, n):
    sigma_t = np.sqrt(omega + alpha * X[t-1]**2)
    X[t] = sigma_t * Z[t]


x = X[:-1]
y_squared = X[1:]**2
variance = omega + alpha * x**2

fig, ax = plt.subplots(1, 3, figsize=(18, 4))

ax[0].plot(X)
ax[0].grid()
ax[0].set_title("Simulated ARCH(1) process")
ax[0].set_xlabel("Time")
ax[0].set_ylabel(r"$X_t$")

ax[1].plot(np.arange(1, len(X)), variance)
ax[1].grid()
ax[1].set_title(r"$\sigma_{t+1}^2 = 0.2 + 0.5X_t^2$")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Conditional variance")

ax[2].set_title("Squared Observations and Conditional Variance")
ax[2].scatter(
    x, y_squared, s=10, alpha=0.2,
    label=r"Observed $X_{t+1}^2$"
)
ax[2].grid()

grid = np.linspace(x.min(), x.max(), 300)
ax[2].plot(
    grid, omega + alpha * grid**2,
    color="black", linewidth=2,
    label="True conditional variance"
)

n_bins = 20
edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
print(edges)
bin_id = np.digitize(x, edges[1:-1])
print(bin_id)

bin_x = [x[bin_id == i].mean() for i in range(n_bins)]
bin_y = [y_squared[bin_id == i].mean() for i in range(n_bins)]

ax[2].scatter(
    bin_x, bin_y, color="red", s=45,
    label="Bin averages", zorder=3
)

#ax[2].set_title("Variance depends on the current observation")
ax[2].set_xlabel(r"$X_t$")
ax[2].set_ylabel(r"$X_{t+1}^2$")
ax[2].legend()

plt.tight_layout()
plt.savefig("arch_plots.png", dpi=200)
plt.show()

def loo_cv(x, y_squared, h):
    distances = x[:, None] - x[None, :]
    weights = np.exp(-0.5 * (distances / h)**2)

    np.fill_diagonal(weights, 0)

    weight_sums = weights.sum(axis=1)
    if np.any(weight_sums == 0):
        return np.inf

    predictions = (weights @ y_squared) / weight_sums

    return np.mean((y_squared - predictions)**2)

plt.figure(figsize=(8, 4))
plt.scatter(x, y_squared, s=10, alpha=0.2, label="Observed $X_{t+1}^2$")
plt.plot(grid, omega + alpha * grid**2, color="black", linewidth=2, label="True conditional variance")
h = np.geomspace(0.05, 0.5, 20)
h = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
cv_errors = []
for best_h in h:
    weights = np.exp( - 0.5 * ((grid[:, None] - x[None, :]) / best_h)**2 )
    variance_est = (weights @ y_squared) / weights.sum(axis=1)
    plt.plot(grid, variance_est, linewidth=2, label=f"h={best_h}")
    loo_error = loo_cv(x, y_squared, best_h)
    cv_errors.append(loo_error)
    print(f"LOO error for h={best_h}: {loo_error:.4f}")

print(f"Best bandwidth: {h[np.argmin(cv_errors)]}, LOO error: {np.min(cv_errors):.4f}")

plt.title("Estimated vs True Conditional Variance")
plt.grid()
plt.xlabel(r"$X_t$")
plt.ylabel(r"$\sigma_{t+1}^2$")

plt.legend(loc="upper right")
plt.tight_layout()

plt.savefig(
    "arch_variance_estimation_LOO.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()
