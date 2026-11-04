import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def simulate_regime_ar(n, a0, a1, sigma, regime_rule, burn=500, seed=1, x0=0.0):
    '''Generic 2-regime AR(1) simulator.
    regime_rule(x_prev, j_prev, rng) -> regime index (0 or 1) at time t.'''
    rng = np.random.default_rng(seed)
    N = n + burn
    x = np.zeros(N); J = np.zeros(N, dtype=int)
    x[0] = x0
    e = rng.standard_normal(N)
    for t in range(1, N):
        j = regime_rule(x[t-1], J[t-1], rng)
        J[t] = j
        x[t] = a0[j] + a1[j] * x[t-1] + sigma[j] * e[t]
    return x[burn:], J[burn:]

def setar_rule(r=0.0):
    return lambda xp, jp, rng: 0 if xp <= r else 1

def conditional_lse(x, J):
    '''Conditional least squares estimation of SETAR(2;1;1).'''
    x_prev = x[:-1]
    x_next = x[1:]
    x0 = x_prev[J[1:] == 0]
    y0 = x_next[J[1:] == 0]
    n0 = len(x0)
    X0 = np.vstack([np.ones(n0), x0]).T
    beta0 = np.linalg.lstsq(X0, y0, rcond=None)[0]


    x1 = x_prev[J[1:] == 1]
    y1 = x_next[J[1:] == 1]
    n1 = len(x1)
    X1 = np.vstack([np.ones(n1), x1]).T
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    
    return beta0, beta1

def loss_function(x, beta0, beta1):
    x_prev = x[:-1]
    x_next = x[1:]
    y_hat = np.where(
        x_prev <= 0,
        2.0 + beta0 * x_prev,
        -1.0 + beta1 * x_prev
    )

    return np.sum((x_next - y_hat) ** 2)




n = 2000
a0 = [2.0, -1.0]
a1 = [0.5, -0.6]
sigma = [1.0, 1.0]

x, J = simulate_regime_ar(n, a0, a1, sigma, setar_rule(r=0.0), seed=1)

res = minimize(
    lambda b: loss_function(x, b[0], b[1]),
    x0=np.array([0.0, 0.0]))

beta0 = res.x[0]
beta1 = res.x[1]

print("Estimated parameters:")
print("Regime 0: Slope = {:.4f}".format(beta0))
print("Regime 1: Slope = {:.4f}".format(beta1))


#beta0, beta1 = conditional_lse(x, J)

estimated_x = np.where(
        x[:-1] <= 0,
        2.0 + beta0 * x[:-1],
        -1.0 + beta1 * x[:-1]
        )
x0_est = np.linspace(min(x), 0, 100)
x1_est = np.linspace(0, max(x), 100)
y0_est = 2.0 + beta0 * x0_est
y1_est = -1.0  + beta1 * x1_est


plt.figure(figsize=(10, 6))
plt.scatter(x[:-1], x[1:], c=J[1:], cmap='coolwarm', alpha=0.5)
plt.axvline(0, color='k', lw=1, ls='--')
plt.plot(x0_est, y0_est, color='blue', lw=2)
plt.plot(x1_est, y1_est, color='red', lw=2)
plt.legend()
plt.title('SETAR(2;1;1) Parameter Estimation')
plt.xlabel('x[t-1]')
plt.ylabel('x[t]')
plt.grid()
plt.savefig('setar_estimated.png')
plt.show()
