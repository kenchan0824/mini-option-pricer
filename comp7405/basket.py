import numpy as np
from scipy.stats import norm


def geom_basket_exact(S1_0, S2_0, K, T, sigma1, sigma2, r, rho, option_type):
    """
    Closed form formula for Geometric basket options

    S1_0: current price for stock 1
    S2_0: current price for stock 2
    K: strike price
    T: time to maturity in years
    sigma1: implied volatility for stock 1
    sigma2: implied volatility for stock 1
    r: risk free interest rate
    rho: correlation coefficient
    option_type: 'C' for call, 'P' for put option

    return: option price
    """

    B0 = np.sqrt(S1_0 * S2_0)

    sigma_B = 0.5 * np.sqrt(sigma1*sigma1 + sigma2*sigma2 + 2*sigma1*sigma2*rho)
    mu_B = r - 0.25 * (sigma1*sigma1 + sigma2*sigma2) + 0.5 * sigma_B*sigma_B

    d1_hat = (np.log(B0/K) + (mu_B + 0.5*sigma_B*sigma_B) * T) / (sigma_B * np.sqrt(T))
    d2_hat = d1_hat - sigma_B * np.sqrt(T)

    df = np.exp(-r*T)
    if option_type == 'C':
        C = df * ( B0*np.exp(mu_B*T)*norm.cdf(d1_hat) - K*norm.cdf(d2_hat) )
        return C

    P = df * ( -B0*np.exp(mu_B*T)*norm.cdf(-d1_hat) + K*norm.cdf(-d2_hat) )
    return P


def geom_basket_payoff(S1_T, S2_T, K, T, r, option_type):
    """
    Discounted Geometric basket option payoff function.

    S1_T: stock 1 price at time T
    S2_T: stock 1 price at time T
    K: strike price
    T: Time to maturity in years
    r: risk free interest rate
    option_type: 'C' for call, 'P' for put

    return: discounted option price
    """
    geom_mean = np.sqrt(S1_T * S2_T)

    if option_type == 'C':
        payoff = np.maximum(geom_mean - K, 0)
    else:
        payoff = np.maximum(K - geom_mean, 0)

    return np.exp(-r * T) * payoff


def arith_basket_payoff(S1_T, S2_T, K, T, r, option_type):
    """
    Discounted Arithmetic basket option payoff function.

    S1_T: stock 1 price at time T
    S2_T: stock 1 price at time T
    K: strike price
    T: Time to maturity in years
    r: risk free interest rate
    option_type: 'C' for call, 'P' for put

    return: discounted option price
    """
    arith_mean = 0.5 * (S1_T + S2_T)

    if option_type == 'C':
        payoff = np.maximum(arith_mean - K, 0)
    else:
        payoff = np.maximum(K - arith_mean, 0)

    return np.exp(-r * T) * payoff


def monte_carlo_basket(S1_0, S2_0, K, T, sigma1, sigma2, r, rho, option_type, sim_type='C',
                      m=100000):
    """
    Monte Carlo Simulation for Geometric / Arithmetic basket options

    S1_0: current price for stock 1
    S2_0: current price for stock 2
    K: strike price
    T: time to maturity in years
    sigma1: implied volatility for stock 1
    sigma2: implied volatility for stock 1
    r: risk free interest rate
    rho: correlation coefficient
    option_type: 'C' for call,
                 'P' for put option
    sim_type: 'G' for geometric basket,
              'A' for arithmetic basket,
              'C' for arithmetic basket with control variate
    m: no. of trials in monte carlo simulation

    return: (mean, confi_upper, confi_lower)
        mean: simulated option price
        confi_upper: 95% confidence interval upper bond
        confi_lower: 95% confidence interval lower bond
    """

    drift1 = np.exp((r - 0.5*sigma1*sigma1) * T)
    drift2 = np.exp((r - 0.5*sigma2*sigma2) * T)

    Z0 = np.random.randn(m)
    Z1 = np.random.randn(m)
    Z2 = rho*Z1 + np.sqrt(1-rho*rho)*Z0

    S1_T = S1_0 * drift1 * np.exp(sigma1 * np.sqrt(T) * Z1)
    S2_T = S2_0 * drift2 * np.exp(sigma2 * np.sqrt(T) * Z2)

    if sim_type == 'G':
        payoff = geom_basket_payoff(S1_T, S2_T, K, T, r, option_type)
        mean = np.mean(payoff)
        std = np.std(payoff)

    elif sim_type == 'A':
        payoff = arith_basket_payoff(S1_T, S2_T, K, T, r, option_type)
        mean = np.mean(payoff)
        std = np.std(payoff)

    else:
        geom_payoff = geom_basket_payoff(S1_T, S2_T, K, T, r, option_type)
        arith_payoff = arith_basket_payoff(S1_T, S2_T, K, T, r, option_type)

        cov_xy = np.mean(arith_payoff * geom_payoff) - np.mean(arith_payoff)*np.mean(geom_payoff)
        theta = cov_xy / np.var(geom_payoff)

        geom_exact = geom_basket_exact(S1_0, S2_0, K, T, sigma1, sigma2, r, rho, option_type)
        Z = arith_payoff + theta * (geom_exact - geom_payoff)

        mean = np.mean(Z)
        std = np.std(Z)

    std_err = 1.96 * std / np.sqrt(m)
    return mean, mean - std_err, mean + std_err

