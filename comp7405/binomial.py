import numpy as np


def binomial_tree(S0, K, T, sigma, r, option_type, N=5):
    """
    Binomial tree method for American call/put options.

    S0: initial stock price
    K: strike price
    T: time to maturity in years
    sigma: annual implied volatility
    r: risk free interest rate
    N: no. of steps
    option_type: 'C' for call, 'P' for put option

    return: American option price
    """

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r*dt) - d) / (u-d)
    df = np.exp(-r*dt)

    # forward pass
    layers = []
    S_out = np.array([S0])
    for i in range(N):
        S_in = S_out
        layers.append(S_in)
        S_out = forward(S_in, u, d)

    # calculate f at the final layer
    if option_type == 'C': # call
        f_out = np.maximum(S_out - K, 0)
    else: # put
        f_out = np.maximum(K - S_out, 0)

    # backward pass
    for i in range(N):
        f_in = backward(f_out, p, df)

        # consider early exercise
        S_in = layers.pop()
        if option_type == 'C': # call
            f_in = np.maximum(S_in - K, f_in)
        else: # put
            f_in = np.maximum(K - S_in, f_in)

        f_out = f_in

    return f_in[0]


def forward(S_in, u, d):
    """
    Compute the forward pass for a single Binomial Tree layer

    S_in: stock prices inputted to the layer, it takes n dimension
    u: upside factor for binomial method
    d: downside factor for binomial method

    return: the (n+1) dimension output stock price for the layer
    """

    n = len(S_in)
    S_out = np.zeros(n+1)
    S_out[:-1] = S_in * u
    S_out[-1] = S_in[-1] * d

    return S_out


def backward(f_out, p, df):
    """
    Compute the backward pass for a single Binomial Tree layer

    f_out: option price for the layer output, it takes (n+1) dimension
    p: probability of upside for binomial method
    df: discount factor for a step

    return: the n dimension option prices that derived backward
    """

    n = len(f_out)
    f_in = np.zeros(n-1)
    f_in = df * (p * f_out[:-1] + (1-p) * f_out[1:])

    return f_in