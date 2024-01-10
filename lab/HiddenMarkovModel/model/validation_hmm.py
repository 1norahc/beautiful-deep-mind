from hidden_markov_model import *
from training_hmm_model import *

def __validation__():
    np.random.seed(42)

    a1 = ProbabilityVector({'1H': 0.7, '2C': 0.3})
    a2 = ProbabilityVector({'1H': 0.4, '2C': 0.6})
    b1 = ProbabilityVector({'1S': 0.1, '2M': 0.4, '3L': 0.5}) 
    b2 = ProbabilityVector({'1S': 0.7, '2M': 0.2, '3L': 0.1})
    A  = ProbabilityMatrix({'1H': a1, '2C': a2})
    B  = ProbabilityMatrix({'1H': b1, '2C': b2})
    pi = ProbabilityVector({'1H': 0.6, '2C': 0.4})

    hmc = HiddenMarkovChain_Uncover(A, B, pi)

    observed_sequence, latent_sequence = hmc.run(5)
    uncovered_sequence = hmc.uncover(observed_sequence)
    """
    |                    | 0   | 1   | 2   | 3   | 4   | 5   |
    |:------------------:|:----|:----|:----|:----|:----|:----|
    | observed sequence  | 3L  | 3M  | 1S  | 3L  | 3L  | 3L  |
    | latent sequence    | 1H  | 2C  | 1H  | 1H  | 2C  | 1H  |
    | uncovered sequence | 1H  | 1H  | 2C  | 1H  | 1H  | 1H  |
    """

    all_possible_states = {'1H', '2C'}
    chain_length = 6  # any int > 0
    all_states_chains = list(product(*(all_possible_states,) * chain_length))

    df = pd.DataFrame(all_states_chains)
    dfp = pd.DataFrame()

    for i in range(chain_length):
        dfp['p' + str(i)] = df.apply(lambda x: 
            hmc.E.df.loc[x[i], observed_sequence[i]], axis=1)

    scores = dfp.sum(axis=1).sort_values(ascending=False)
    df = df.iloc[scores.index]
    df['score'] = scores
    df.head(10).reset_index()
    """
    |    index | 0   | 1   | 2   | 3   | 4   | 5   |   score |
    |:--------:|:----|:----|:----|:----|:----|:----|--------:|
    |        8 | 1H  | 1H  | 2C  | 1H  | 1H  | 1H  |     3.1 |
    |       24 | 1H  | 2C  | 2C  | 1H  | 1H  | 1H  |     2.9 |
    |       40 | 2C  | 1H  | 2C  | 1H  | 1H  | 1H  |     2.7 |
    |       12 | 1H  | 1H  | 2C  | 2C  | 1H  | 1H  |     2.7 |
    |       10 | 1H  | 1H  | 2C  | 1H  | 2C  | 1H  |     2.7 |
    |        9 | 1H  | 1H  | 2C  | 1H  | 1H  | 2C  |     2.7 |
    |       25 | 1H  | 2C  | 2C  | 1H  | 1H  | 2C  |     2.5 |
    |        0 | 1H  | 1H  | 1H  | 1H  | 1H  | 1H  |     2.5 |
    |       26 | 1H  | 2C  | 2C  | 1H  | 2C  | 1H  |     2.5 |
    |       28 | 1H  | 2C  | 2C  | 2C  | 1H  | 1H  |     2.5 |
    """

    dfc = df.copy().reset_index()
    for i in range(chain_length):
        dfc = dfc[dfc[i] == latent_sequence[i]]
    """
    |   index | 0   | 1   | 2   | 3   | 4   | 5   |   score |
    |:-------:|:----|:----|:----|:----|:----|:----|--------:|
    |      18 | 1H  | 2C  | 1H  | 1H  | 2C  | 1H  |     1.9 |
    """

    np.random.seed(42)

    observations = ['3L', '2M', '1S', '3L', '3L', '3L']

    states = ['1H', '2C']
    observables = ['1S', '2M', '3L']

    hml = HiddenMarkovLayer.initialize(states, observables)
    hmm = HiddenMarkovModel(hml)

    hmm.train(observations, 25)
__validation__()


