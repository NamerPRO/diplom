import numpy as np

from core.acoustic_model.gmm import GMM


class HMMState:
    """
    Represents a Hidden Markov Model single hidden state.
    """

    def __init__(self, name: str, gmm: GMM, initial_probability: np.float64, is_word_final_state=False, log_probability: bool = True):
        """
        Args:
            name (str): The unique name of the Hidden Markov Model state.
                Example: [y-eh+l].

            gmm (GMM): The Gaussian Mixture model assosiated with this state.

            initial_probability (np.float64): The probability of starting in this state.
                Sum of such probabilities for all HMM states must equal to 1.

            is_word_final_state (bool): Whether this Hidden Markov Model state is a word final state.
                Silence state is also considered a word final state.

            use_log_probability (bool, optional): Whether to use log-probability or not.
                Defaults to True.
        """
        self.name = name
        self.gmm = gmm
        self.initial_probability = initial_probability if log_probability else np.log(initial_probability)
        self.is_word_final_state = is_word_final_state
