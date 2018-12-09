'''

The goal of this program is trying to train a DQN
to trade on basic type of stationary signals

The strategy is based on mean reversion


'''


import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from numpy.core.multiarray import ndarray
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt


def generate_data(mu, sigma, T):
    p = list()
    for t in range(T):
        p.append(np.random.normal(mu, sigma))
    ret = np.array(p)
    price = np.cumsum(ret)

    return price

def display(signal):
    plt.plot(signal)
    plt.plot(np.diff(signal))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(['Price', 'Diffs'])
    plt.show()

x = generate_data(0,1,100)
display(x)