import numpy as np
from scipy.stats import norm


def geom_asian_exact(S0, K, T, sigma, r, n, option_type):
    """
    Closed form formula for Geometric Asian options

    S0: current stock price
    K: strike price
    T: time to maturity in years
    sigma: implied volatility
    r: risk free interest rate
    n: no. of observation times
    option_type: 'C' for call, 'P' for put option

    return: option price
    """

    sigma_hat = sigma * np.sqrt( (n+1) * (2*n+1) / (6*n*n) )
    mu_hat = (r-0.5*sigma*sigma)*(n+1)/(2*n) + 0.5*sigma_hat*sigma_hat

    d1 = (np.log(S0 / K) + (mu_hat + 0.5 * sigma_hat * sigma_hat) * T) / (sigma_hat * np.sqrt(T))
    d2 = d1 - sigma_hat * np.sqrt(T)

    df = np.exp(-r*T)
    if option_type == 'C':
        C = df * ( S0*np.exp(mu_hat*T)*norm.cdf(d1) - K*norm.cdf(d2) )
        return C

    P = df * ( -S0*np.exp(mu_hat*T)*norm.cdf(-d1) + K*norm.cdf(-d2) )
    return P


def geom_asian_payoff(S_path, K, T, r, option_type):
    """
    Discounted Geometric Asian option payoff function.

    S_path: stock prices at n discrete time point
    K: strike price
    T: Time to maturity in years
    r: risk free interest rate
    option_type: 'C' for call, 'P' for put

    return: discounted option price
    """
    n = len(S_path)
    geom_mean = np.exp( 1.0 / n * np.sum(np.log(S_path), axis=0) )

    if option_type == 'C':
        payoff = np.maximum(geom_mean - K, 0)
    else:
        payoff = np.maximum(K - geom_mean, 0)

    return np.exp(-r * T) * payoff


def arith_asian_payoff(S_path, K, T, r, option_type):
    """
    Discounted Arithmetic Asian option payoff function.

    S_path: stock prices at n discrete time point
    K: strike price
    T: Time to maturity in years
    r: risk free interest rate
    option_type: 'C' for call, 'P' for put

    return: discounted option price
    """
    n = len(S_path)
    arith_mean = np.mean(S_path, axis=0)

    if option_type == 'C':
        payoff = np.maximum(arith_mean - K, 0)
    else:
        payoff = np.maximum(K - arith_mean, 0)

    return np.exp(-r * T) * payoff


def monte_carlo_asian(S0, K, T, sigma, r, n, option_type, sim_type='C', m=100000):
    """
    Monte Carlo Simulation for Geometric / Arithmetic Asian options

    S0: current stock price
    K: strike price
    T: time to maturity in years
    sigma: annual volatility
    r: risk free interest rate
    n: no. of observation times
    option_type: 'C' for call,
                 'P' for put option
    sim_type: 'G' for geometric asian,
              'A' for arithmetic asian,
              'C' for arithmetic asian with control variate
    m: no. of trials in monte carlo simulation

    return: (mean, confi_upper, confi_lower)
        mean: simulated option price
        confi_upper: 95% confidence interval upper bond
        confi_lower: 95% confidence interval lower bond
    """

    S_path = np.zeros((m, n))

    dt = T / n
    drift = np.exp((r - 0.5*sigma*sigma) * dt)
    Z = np.random.randn(m)
    S_path[:, 0] = S0 * drift * np.exp(sigma * np.sqrt(dt) * Z)
    for i in range(1, n):
        Z = np.random.randn(m)
        S_path[:, i] = S_path[:, i-1] * drift * np.exp(sigma * np.sqrt(dt) * Z)

    if sim_type == 'G':
        #print(S_path.shape)
        payoff = geom_asian_payoff(S_path.T, K, T, r, option_type)
        #print(payoff.shape)
        mean = np.mean(payoff)
        std = np.std(payoff)

    elif sim_type == 'A':
        payoff = arith_asian_payoff(S_path.T, K, T, r, option_type)
        mean = np.mean(payoff)
        std = np.std(payoff)

    else:
        geom_payoff = geom_asian_payoff(S_path.T, K, T, r, option_type)
        arith_payoff = arith_asian_payoff(S_path.T, K, T, r, option_type)

        cov_xy = np.mean(arith_payoff * geom_payoff) - np.mean(arith_payoff)*np.mean(geom_payoff)
        theta = cov_xy / np.var(geom_payoff)

        geom_exact = geom_asian_exact(S0, K, T, sigma, r, n, option_type)
        payoff = arith_payoff + theta * (geom_exact - geom_payoff)

        mean = np.mean(payoff)
        std = np.std(payoff)

    std_err = 1.96 * std / np.sqrt(m)
    return mean, mean - std_err, mean + std_err