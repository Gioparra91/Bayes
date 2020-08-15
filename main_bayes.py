# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:41:28 2020

@author: gparrav
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

os.chdir(r'C:\Users\gparrav\OneDrive - MORNINGSTAR INC\Desktop\Projects\Bayes Theorem Investing')

# %% Settings

outcome = ['T','H']
iterations = 10
df = pd.DataFrame()
prior = 0.9
green_coin_head = 0.3
red_coin_head = 1-green_coin_head

# %% Framework

# H or T
df['RndGen_HorT'] = np.random.choice(outcome, size=iterations)

# 1 or 0
df['RndGen_1or0'] = df['RndGen_HorT']
for i in range(df.shape[0]):
    if df['RndGen_HorT'].iloc[i]=='T':
        df['RndGen_1or0'].iloc[i] = 0
    else:
        df['RndGen_1or0'].iloc[i] = 1
    
# cum sum
df['SumWin'] = df['RndGen_1or0'].cumsum()

# Prior/confidence/initial belief of an event (that the coin is green)
# marginal
df['p(H)'] = prior

# Anti prior
# marginal
df['p(~H)'] = 1-prior

# Overlap area P(H|D) = the proba of the event happen within p(H) prior given that
# the data showed the event happened within p(D) is the overlapping area p(H&D) divided 
# or normalized by the area of p(D).

# Believe it's a green coing. gives event H with 30%p and T with 70%p
# Think of it like a binomial tree that evolve in time TTTTTT is on branch, TTTHTTT is another
# conditional. "|H" means given all the previous values x0....xn

# P(H|F)=(n x)*θ^x(1−θ)^(n−x)=(1 1)*0.5^1*(0.5)^0=0.5
df['n'] = range(1, iterations+1)
df['x'] = range(1, iterations+1)

df['p(D|H)'] = green_coin_head**df['x'] * (1 - green_coin_head)**(df['n'] - df['x'])
df['p(D|~H)'] = red_coin_head**df['x'] * (1 - red_coin_head)**(df['n'] - df['x'])



# Normalizer probability that ensure proba space is 1. It updates with new data
# marginal
df['p(D)'] = df['p(D|H)'] * df['p(H)'] + df['p(D|~H)'] * df['p(~H)']

# second order updated probability of the event to happen condition on new data
# this is what we are interested in. it's also a conditional
df['p(H|D'] = df['p(D|H)'] * df['p(H)'] / df['p(D)']





import numpy as np
import matplotlib.pyplot as plt


N = 100 # Number of flips
BIAS_HEADS = 0.3 # The bias of the coin

bias_range = np.linspace(0, 1, 101) # The range of possible biases
prior_bias_heads = np.ones(len(bias_range)) / len(bias_range) # Uniform prior distribution
flip_series = (np.random.rand(N) <= BIAS_HEADS).astype(int) # A series of N 0's and 1's (coin flips)

for flip in flip_series:
    likelihood = bias_range**flip * (1-bias_range)**(1-flip) # p(D|H)
    evidence = np.sum(likelihood * prior_bias_heads) # p(D) normalizer that ensure space=1
    prior_bias_heads = likelihood * prior_bias_heads / evidence # 2nd order proba or posterior

plt.plot(bias_range, prior_bias_heads)
plt.xlabel('Heads Bias')
plt.ylabel('P(Heads Bias)')
plt.grid()
plt.show()










# on probabililities
def event_probability(event_outcomes, sample_space):
    probability = (event_outcomes / sample_space) * 100
    return round(probability, 1)

cards = 52
queen_of_spade = 1
event_probability(queen_of_spade, cards)

# the sample space it's not always easy to compute: permutations and combinations

