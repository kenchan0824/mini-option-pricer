import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, sigma, r, q, option_type):
    """
    Black-Scholes Formula with Repo Rate

    S: stock price
    K: strike price
    T: time to maturity in years
    sigma: annual implied volatility
    r: risk free interest rate
    q: repo rate
    option_type: 'C' for call, 'P' for put option

    return: option price
    """

    d1 = (np.log(S / K) + (r - q) * T) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)

    C = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option_type == 'C':
        return C

    P = C + K * np.exp(-r * T) - S * np.exp(-q * T)
    return P


def cal_vega(S, K, T, sigma, r, q):
    """
    Derivative of Black-Scholes Formula w.r.t. sigma

    S: stock price
    K: strike price
    T: time to maturity in years
    sigma: annual implied volatility
    r: risk free rate
    q: repo rate

    return: vega - the gradient of C or P along sigma
    """
    d1 = (np.log(S / K) + (r - q) * T) / (sigma * np.sqrt(T)) + 0.5 * sigma * np.sqrt(T)
    vega = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)

    return vega


def imp_vol(V, S, K, T, r, q, option_type):
    """
    Calculate put option implied volatility using newton's method

    V: option market price
    type: 'C' for call, 'P' for put
    S: stock price
    K: strike price
    T: time to maturity in years
    r: risk free rate
    q: repo rate
    option_type: 'C' for call, 'P' for put option

    return: sigma - implied volatility
    """

    epsilon = 1e-8

    # initial sigma
    sigma = np.sqrt(2.0 * np.abs(np.log(S / K) / T + (r - q)))

    for i in range(0, 1000):
        # with sigma, get the theoretical option price
        V_sigma = black_scholes(S, K, T, sigma, r, q, option_type)

        # vega is f'(x)
        vega = cal_vega(S, K, T, sigma, r, q)
        if vega == 0:
            return np.nan

        # update x by x - f(x) / f'(x)
        increment = (V_sigma - V) / vega
        sigma -= increment

        if np.abs(increment) < epsilon:
            break

    return sigma
