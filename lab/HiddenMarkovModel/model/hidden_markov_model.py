# Hidden Markov models
# =======================


"""
NOTE:
T - length of the observation sequence.
N - number of latent (hidden) states.
M - number of observables.
Q = {q₀, q₁, …} - hidden states.
V = {0, 1, …, M — 1} - set of possible observations.
A - state transition matrix.
B - emission probability matrix.
π- initial state probability distribution.
O - observation sequence.
X = (x₀, x₁, …), x_t ∈ Q - hidden state sequence.

Having that set defined, we can calculate the probability of any state and observation using the matrices:

A = {a_ij} — begin an transition matrix.
B = {b_j(k)} — being an emission matrix.

The probabilities associated with transition and observation (emission) are: [PV/fig1],[PV/fig2]
The model is therefore defined as a collection: [PV/fig3]
====
"""

import numpy as np
import pandas as pd

# PROBABILITY VECTOR
# -----------------------

class ProbabilityVector:

    """
    NOTE:
    Note that when e.g. multiplying a PV with a scalar, the returned structure is a resulting numpy array, not another PV. This is 
    because multiplying by anything other than 1 would violate the integrity of the PV itself.
    Internally, the values are stored as a numpy array of size (1 × N).
    [PV/fig4.png]
    ====
    """
        
    def __init__(self, probabilities: dict):
        states, probs = probabilities.keys(), probabilities.values()

        assert len(states) == len(probs), "The probabilities must match the states."
        assert len(states) == len(set(states)), "The states must be unique."
        assert abs(sum(probs) - 1.0) < 1e-12, "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), "Probabilities must be numbers from [0,1] interval."

        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x: probabilities[x], self.states))).reshape(1, -1)

    def __str__(self):
        return """
                The PV objects need to satisfy the following mathematical operations (for the purpose of constructing of HMM):

                1. comparison (__eq__) - to know if any two PV's are equal,
                2. element-wise multiplication of two PV's or multiplication with a scalar (__mul__ and __rmul__).
                3. dot product (__matmul__) - to perform vector-matrix multiplication
                4. division by number (__truediv__),
                5. argmax to find for which state the probability is the highest.
                6. __getitem__ to enable selecting value by the key.

                """

    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, state: list):
        return cls(dict(zip(states, list(array))))
    
    @property
    def dict(self):
        return {k : v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, 
                            columns=self.states, 
                            index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)
    
    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False
    
    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotADirectoryError
    
    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)
    
    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values
    
    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)
    
    def argmax(self):
        index = self.values.argmax()
        return self.states[index]

# TEST 1
"""a1 = ProbabilityVector({'rain': 0.7, 'sun': 0.3})
a2 = ProbabilityVector({'sun': 0.1, 'rain': 0.9})
print(a1.df)
print(a2.df)

print("Comparison:", a1 == a2)
print("Element-wise multiplication:", a1 * a2)
print("Argmax:", a1.argmax())
print("Getitem:", a1['rain'])
"""


# PROBABILITY MATRIX
# -----------------------

class ProbabilityMatrix:
    """
    NOTE:
    Here, the way we instantiate PM's is by supplying a dictionary of PV's to the constructor of the class. By doing this, we not only ensure that every row of PM is stochastic, but also supply the names for every observable.

    Our PM can, therefore, give an array of coefficients for any observable. 
    Mathematically, the PM is a matrix:
    
    [PM_fig1.png]

    The other methods are implemented in similar way to PV.
    """
    def __init__(self, prob_vec_dict: dict):
        
        assert len(prob_vec_dict) > 1, "The number of input probabilities vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), "All observables must be unique."

        self.states = sorted(prob_vec_dict) 
        self.observables = prob_vec_dict[self.states[0]].states
        self.values = np.stack([prob_vec_dict[x].values for x in self.states]).squeeze()
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list, observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x ))) for x in array]
        return cls(dict(zip(states, p_vecs)))
    
    @property
    def dict(self):
        return self.df.to_dict()
    
    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.observables, index=self.states)
    
    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(self.values.shape, self.states, self.observables)
    
    def __getitem__(self, observables: str) -> np.ndarray:
        if observables not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observables)
        return self.values[:, index].reshape(-1, 1)

# EXAMPLE PV and PM
# TEST 2
"""a1 = ProbabilityVector({'rain': 0.7, 'sun': 0.3})
a2 = ProbabilityVector({'rain': 0.6, 'sun': 0.4})
A  = ProbabilityMatrix({'hot': a1, 'cold': a2})

print(A)
print(A.df)
>>> PM (2, 2) states: ['cold', 'hot'] -> obs: ['rain', 'sun'].
>>>      rain  sun
   cold   0.6  0.4
   hot    0.7  0.3

b1 = ProbabilityVector({'0S': 0.1, '1M': 0.4, '2L': 0.5})
b2 = ProbabilityVector({'0S': 0.7, '1M': 0.2, '2L': 0.1})
B =  ProbabilityMatrix({'0H': b1, '1C': b2})

print(B)
print(B.df)
>>> PM (2, 3) states: ['0H', '1C'] -> obs: ['0S', '1M', '2L'].
>>>       0S   1M   2L
     0H  0.1  0.4  0.5
     1C  0.7  0.2  0.1

P = ProbabilityMatrix.initialize(list('abcd'), list('xyz'))
print('Dot product:', a1 @ A)
print('Initialization:', P)
print(P.df)
>>> Dot product: [[0.63 0.37]]
>>> Initialization: PM (4, 3) 
    states: ['a', 'b', 'c', 'd'] -> obs: ['x', 'y', 'z'].
>>>          x         y         z
   a  0.323803  0.327106  0.349091
   b  0.318166  0.326356  0.355478
   c  0.311833  0.347983  0.340185
   d  0.337223  0.316850  0.345927"""


# HIDDEN MARKOV CHAIN
# -----------------------

""" 
NOTE:
Before we proceed with calculating the score, let's use our PV and PM definitions to implement the Hidden Markov Chain.

Again, we will do so as a class, calling it HiddenMarkovChain. It will collate at A, B and π. Later on, we will implement more methods that are applicable to this class.

Computing score
Computing the score means to find what is the probability of a particular chain of observations O given our (known) model λ = (A, B, π). In other words, we are interested in finding p(O|λ).

We can find p(O|λ) by marginalizing all possible chains of the hidden variables X, where X = {x₀, x₁, …}: [HMC/fig1]

Since p(O|X, λ) = ∏ b(O) (the product of all probabilities related to the observables) and 
p(X|λ)=π ∏ a (the product of all probabilities of transitioning from x at t to x at t + 1, 
the probability we are looking for (the score) is: [HMC/fig2]
====
"""

from itertools import product
from functools import reduce

class HiddenMarkovChain:
    
    def __init__(self, T, E, pi):
        self.T = T # transimission matrix A 
        self.E = E # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables
    
    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(len(self.states), len(self.observables))
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, observables)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi) 
    
    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def score(self, observations: list) -> float:
        def mul(x, y): return x * y
    
        score = 0 
        all_chains = self._create_all_chains(chain_length=len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))

            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]

            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score
    
# TEST 3
"""a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})

b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5})
b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})

A = ProbabilityMatrix({'1H': a1, '2C': a2})
B = ProbabilityMatrix({'1H': b1, '2C': b2})
pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

hmc = HiddenMarkovChain(A, B, pi)
observations = ['1S', '2M', '3L', '2M', '1S']

print("Score for {} is {:f}.".format(observations, hmc.score(observations)))

NOTE: 
If our implementation is correct, then all score values for all possible observation chains, 
for a given model should add up to one. Namely: [HMC/fig3]
====

all_possible_observations = {'1S', '2M', '3L'}
chain_length = 3  # any int > 0
all_observation_chains = list(product(*(all_possible_observations,) * chain_length))
all_possible_scores = list(map(lambda obs: hmc.score(obs), all_observation_chains))
print("All possible scores added: {}.".format(sum(all_possible_scores)))

PTEST 1 -> result: 
>>> All possible scores added: 0.9999999999999998.
PTEST 2 -> result: 
>>> All possible scores added: 1.0.
"""

# SCORE WITH FORWARD-PASS 
# -----------------------

"""
NOTE: 
Computing the score the way we did above is kind of naive. In order to find the number 
for a particular observation chain O, we have to compute the score for all possible latent 
variable sequences X. That requires 2TN^T multiplications, which even for small numbers takes time.

Another way to do it is to calculate partial observations of a sequence up to time t.

For and i ∈ {0, 1, …, N-1} and t ∈ {0, 1, …, T-1} : [FP/fig1]

Consequently,
[FP/fig2]

and
[FP/fig3]

Then
[FP/fig4]

Note that α_t is a vector of length N. The sum of the product α a can, in fact, 
be written as a dot product. Therefore: 
[FP/fig5]

where by the star, we denote an element-wise multiplication.

With this implementation, we reduce the number of multiplication to N²T and can take advantage of 
vectorization.
"""

class HiddenMarkovChain_FP(HiddenMarkovChain):

    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t-1, :].reshape(1, -1) @ self.T.values) * self.E[observations[t]].T
        return alphas
    
    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())

# TEST 4 m. Example1
"""
hmc_fp = HiddenMarkovChain_FP(A, B, pi)

observations = ['1S', '2M', '3L', '2M', '1S']
print("Score for {} is {:f}.".format(observations, hmc_fp.score(observations)))
>>> All possible scores added: 1.0.
"""


# HIDDEN MARKOV CHAIN - SIMULATION 
# -----------------------

class HiddenMarkovChain_Simulation(HiddenMarkovChain):
    
    def run(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be non-negative number."
        
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)

        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())

        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs =  prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())
            
        return o_history, s_history

# TEST 5
"""hmc_s = HiddenMarkovChain_Simulation(A, B, pi)
observation_hist, states_hist = hmc_s.run(100)  # length = 100
stats = pd.DataFrame({
    'observations': observation_hist,
    'states': states_hist}).applymap(lambda x: int(x[0])).plot()"""


""" 
NOTE: Laten states
The state matrix A is given by the following coefficients: 
[laten-states/fig1]
[laten-states/fig2]
[laten-states/fig3]
[laten-states/fig4]

Consequently, the probability of “being” in the state “1H” at t+1, 
regardless of the previous state, is equal to:
[laten-states/fig5]

If we assume that the prior probabilities of being at some state at are totally random, 
then p(1H) = 1 and p(2C) = 0.9, which after renormalizing give 0.55 and 0.45, respectively.

If we count the number of occurrences of each state and divide it by the number of elements in our sequence, 
we would get closer and closer to these number as the length of the sequence grows.

EXAMPLE
hmc_s = HiddenMarkovChain_Simulation(A, B, pi)

stats = {}
for length in np.logspace(1, 5, 40).astype(int):
    observation_hist, states_hist = hmc_s.run(length)
    stats[length] = pd.DataFrame({
        'observations': observation_hist,
        'states': states_hist}).applymap(lambda x: int(x[0]))

S = np.array(list(map(lambda x: 
        x['states'].value_counts().to_numpy() / len(x), stats.values())))

plt.semilogx(np.logspace(1, 5, 40).astype(int), S)
plt.xlabel('Chain length T')
plt.ylabel('Probability')
plt.title('Converging probabilities.')
plt.legend(['1H', '2C'])
plt.show()
"""

# EXPANDING THE CLASS
"""
NOTE: 
We have defined α to be the probability of partial observation of the sequence up to time .
[expanding-class/fig1]

Now, let's define the “opposite” probability. Namely, the probability of observing the sequence from T - 1down to t.

For t= 0, 1, …, T-1 and i=0, 1, …, N-1, we define:
[expanding-class/fig2]

c`1As before, we can β(i) calculate recursively:
[expanding-class/fig3]

Then for t ≠ T-1:
[expanding-class/fig4]

which in vectorized form, will be:
[expanding-class/fig5]

Finally, we also define a new quantity γ to indicate the state q_i at time t, 
for which the probability (calculated forwards and backwards) is the maximum:
[expanding-class/fig6]

Consequently, for any step t = 0, 1, …, T-1, the state of the maximum likelihood can be found using:
[expanding-class/fig7]
"""

class HiddenMarkovChain_Uncover(HiddenMarkovChain_Simulation):

    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) * self.E[observations[t]].T
        
        return alphas

    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) -2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)

        return betas 

    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs)) 
