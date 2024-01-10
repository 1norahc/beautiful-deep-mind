# Training hiden Markov model 
# ===========================

from hidden_markov_model import *

# TRAINING THE MODEL
"""
NOTE: 
The time has come to show the training procedure. Formally, 
we are interested in finding λ = (A, B, π) such that given a desired observation sequence O, 
our model λ would give the best fit.

#Expanding class
Here, our starting point will be the `HiddenMarkovModel_Uncover` 
that we have defined earlier. We will add new methods to train it.

Knowing our latent states Q and possible observation states O, 
we automatically know the sizes of the matrices A and B, hence N and M. 
However, we need to determine a and b and π.

For t = 0, 1, …, T-2 and i, j =0, 1, …, N -1, we define “di-gammas”:
[training-expanding-class/fig1]

γ(i, j) is the probability of transitioning for q at t to t + 1. 
Writing it in terms of α, β, A, B we have:
[training-expanding-class/fig2]

Now, thinking in terms of implementation, we want to avoid looping over i, j and t at the same time, 
as it’s gonna be deadly slow. Fortunately, we can vectorize the equation:
[training-expanding-class/fig3]

Having the equation for γ(i, j), we can calculate
[training-expanding-class/fig4]

To find λ = (A, B, π), we do 
For i = 0, 1, …, N-1:
[training-expanding-class/fig5]

or

[training-expanding-class/fig6]

For i, j = 0, 1, …, N-1:
[training-expanding-class/fig7]

For j = 0, 1, …, N-1 and k = 0, 1, …, M-1:
[training-expanding-class/fig8]
"""

from hidden_markov_model import HiddenMarkovChain_Uncover, \
                                ProbabilityVector, \
                                ProbabilityMatrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class HiddenMarkovLayer(HiddenMarkovChain_Uncover):
    
    def _digammas(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states) 
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        
        return digammas

"""
NOTE:
Having the “layer” supplemented with the ._difammas method, we should be able to perform all the necessary calculations. However, it makes sense to delegate the "management" of the layer to another class. In fact, the model training can be summarized as follows:

1. Initialize A, B and π.
2. Calculate γ(i, j).
3. Update the model’s A, B and π.
4. We repeat the 2. and 3. until the score p(O|λ) no longer increases.
"""

class HiddenMarkovModel:

    def __init__(self, hml: HiddenMarkovLayer):
        self.layer = hml
        self._score_init = 0
        self.score_history = list()
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(states, observables)
        return cls(layer)
    
    def update(self, observations: list) -> float:
        alpha = self.layer._alphas(observations)
        beta = self.layer._betas(observations)
        digamma = self._digammas(observations)
        score = alpha[-1].sum()
        gamma = alpha * beta / score

        L = len(alpha)
        obs_idx = [self.layer.observables.index(x) for x in observations]
        capture = np.zeros((L, len(self.layer.states), len(self.layer.observables)))
        for t in range(L):
            capture[t, :, obs_idx[t]] = 1.0
        
        pi = gamma[0]
        T = digamma.sum(axis=0) / gamma[:-1].sum(axis=0).reshape(-1, 1)
        E = (capture * gamma[:, :, np.newaxis]).sum(axis=0) / gamma.sum(axis=0).reshape(-1, 1)

        self.layer.pi = ProbabilityVector.from_numpy(pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.from_numpy(T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.from_numpy(E, self.layer.states, self.layer.observables)

        return score
    
    def train(self, observations: list, epochs: int, tol=None):
        self._score_init = 0
        self.score_history = (epochs + 1) * [0]
        early_stopping = isinstance(tol, (int, float))

        for epoch in range(1, epochs + 1):
            score = self.update(observations)
            print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))

            if early_stopping and abs(self._score_init - score) / score < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch] = score

# TEST_TRAIN 1 
"""np.random.seed(42)

observations = ['3L', '2M', '1S', '3L', '3L', '3L']

states = ['1H', '2C']
observables = ['1S', '2M', '3L']

hml = HiddenMarkovLayer.initialize(states, observables)
hmm = HiddenMarkovModel(hml)

hmm.train(observations, 25)

# VERIFICATION
|    | 0   | 1   | 2   | 3   | 4   | 5   |
|---:|:----|:----|:----|:----|:----|:----|
|  0 | 3L  | 2M  | 1S  | 3L  | 3L  | 3L  |
RUNS = 100000
T = 5

chains = RUNS * [0]
for i in range(len(chains)):
    chain = hmm.layer.run(T)[0]
    chains[i] = '-'.join(chain)
df = pd.DataFrame(pd.Series(chains).value_counts(), columns=['counts']).reset_index().rename(columns={'index': 'chain'})
df = pd.merge(df, df['chain'].str.split('-', expand=True), left_index=True, right_index=True)

s = []
for i in range(T + 1):
    s.append(df.apply(lambda x: x[i] == observations[i], axis=1))

df['matched'] = pd.concat(s, axis=1).sum(axis=1)
df['counts'] = df['counts'] / RUNS * 100
df = df.drop(columns=['chain'])
df.head(30)
---
|---:|---------:|:----|:----|:----|:----|:----|:----|----------:|
|  0 |    8.907 | 3L  | 3L  | 3L  | 3L  | 3L  | 3L  |         4 |
|  1 |    4.422 | 3L  | 2M  | 3L  | 3L  | 3L  | 3L  |         5 |
|  2 |    4.286 | 1S  | 3L  | 3L  | 3L  | 3L  | 3L  |         3 |
|  3 |    4.284 | 3L  | 3L  | 3L  | 3L  | 3L  | 2M  |         3 |
|  4 |    4.278 | 3L  | 3L  | 3L  | 2M  | 3L  | 3L  |         3 |
|  5 |    4.227 | 3L  | 3L  | 1S  | 3L  | 3L  | 3L  |         5 |
|  6 |    4.179 | 3L  | 3L  | 3L  | 3L  | 1S  | 3L  |         3 |
|  7 |    2.179 | 3L  | 2M  | 3L  | 2M  | 3L  | 3L  |         4 |
|  8 |    2.173 | 3L  | 2M  | 3L  | 3L  | 1S  | 3L  |         4 |
|  9 |    2.165 | 1S  | 3L  | 1S  | 3L  | 3L  | 3L  |         4 |
| 10 |    2.147 | 3L  | 2M  | 3L  | 3L  | 3L  | 2M  |         4 |
| 11 |    2.136 | 3L  | 3L  | 3L  | 2M  | 3L  | 2M  |         2 |
| 12 |    2.121 | 3L  | 2M  | 1S  | 3L  | 3L  | 3L  |         6 |
| 13 |    2.111 | 1S  | 3L  | 3L  | 2M  | 3L  | 3L  |         2 |
| 14 |    2.1   | 1S  | 2M  | 3L  | 3L  | 3L  | 3L  |         4 |
| 15 |    2.075 | 3L  | 3L  | 3L  | 2M  | 1S  | 3L  |         2 |
| 16 |    2.05  | 1S  | 3L  | 3L  | 3L  | 3L  | 2M  |         2 |
| 17 |    2.04  | 3L  | 3L  | 1S  | 3L  | 3L  | 2M  |         4 |
| 18 |    2.038 | 3L  | 3L  | 1S  | 2M  | 3L  | 3L  |         4 |
| 19 |    2.022 | 3L  | 3L  | 1S  | 3L  | 1S  | 3L  |         4 |
| 20 |    2.008 | 1S  | 3L  | 3L  | 3L  | 1S  | 3L  |         2 |
| 21 |    1.955 | 3L  | 3L  | 3L  | 3L  | 1S  | 2M  |         2 |
| 22 |    1.079 | 1S  | 2M  | 3L  | 2M  | 3L  | 3L  |         3 |
| 23 |    1.077 | 1S  | 2M  | 3L  | 3L  | 3L  | 2M  |         3 |
| 24 |    1.075 | 3L  | 2M  | 1S  | 2M  | 3L  | 3L  |         5 |
| 25 |    1.064 | 1S  | 2M  | 1S  | 3L  | 3L  | 3L  |         5 |
| 26 |    1.052 | 1S  | 2M  | 3L  | 3L  | 1S  | 3L  |         3 |
| 27 |    1.048 | 3L  | 2M  | 3L  | 2M  | 1S  | 3L  |         3 |
| 28 |    1.032 | 1S  | 3L  | 1S  | 2M  | 3L  | 3L  |         3 |
| 29 |    1.024 | 1S  | 3L  | 1S  | 3L  | 1S  | 3L  |         3 |

And here are the sequences that we don't want the model to create.

|     |   counts | 0   | 1   | 2   | 3   | 4   | 5   |   matched |
|----:|---------:|:----|:----|:----|:----|:----|:----|----------:|
| 266 |    0.001 | 1S  | 1S  | 3L  | 3L  | 2M  | 2M  |         1 |
| 267 |    0.001 | 1S  | 2M  | 2M  | 3L  | 2M  | 2M  |         2 |
| 268 |    0.001 | 3L  | 1S  | 1S  | 3L  | 1S  | 1S  |         3 |
| 269 |    0.001 | 3L  | 3L  | 3L  | 1S  | 2M  | 2M  |         1 |
| 270 |    0.001 | 3L  | 1S  | 3L  | 1S  | 1S  | 3L  |         2 |
| 271 |    0.001 | 1S  | 3L  | 2M  | 1S  | 1S  | 3L  |         1 |
| 272 |    0.001 | 3L  | 2M  | 2M  | 3L  | 3L  | 1S  |         4 |
| 273 |    0.001 | 1S  | 3L  | 3L  | 1S  | 1S  | 1S  |         0 |
| 274 |    0.001 | 3L  | 1S  | 2M  | 2M  | 1S  | 2M  |         1 |
| 275 |    0.001 | 3L  | 3L  | 2M  | 1S  | 3L  | 2M  |         2 |
"""