import numpy as np

from recognition.models.acoustic.hmm import HMMManager


# test = HMMManager.get_phone_level_hmm('и', r'C:\Users\PeterA\Desktop\vkr\test\_____attempt14\sounds')
# test2 = HMMManager.get_phone_level_hmm('и', r'C:\Users\PeterA\Desktop\vkr\test\_____attempt14\sounds')
#
# print(test.transitions)
# print([test.states[i].initial_probability for i in range(test.states_n)])
#
#
# print(test2.transitions)
# print([test2.states[i].initial_probability for i in range(test2.states_n)])


# word_hmm = HMMManager.create_word_level_hmm(
#     word='включить',
#     lexicon=['ф', 'к', 'лю', 'ч', 'и', 'ть'],
#     gmm_dataset_path=r'C:\Users\PeterA\Desktop\vkr\test\_____attempt14\sounds'
# )
# np.set_printoptions(linewidth=np.inf)
# print(word_hmm.transitions)
# print([word_hmm.states[i].initial_probability for i in range(word_hmm.states_n)])


lexicon = {
    'включить' : ['ф', 'к', 'лю', 'ч', 'и', 'ть'],
    'лампу' : ['ла', 'м', 'п', 'у']
}
sentence_hmm = HMMManager.create_sentence_level_hmm(
    sentence='включить лампу',
    full_lexicon=lexicon,
    gmm_dataset_path=r'C:\Users\PeterA\Desktop\vkr\test\_____attempt14\sounds'
)
np.set_printoptions(linewidth=np.inf)
print(sentence_hmm.transitions)
print([sentence_hmm.states[i].initial_probability for i in range(sentence_hmm.states_n)])

HMMManager.update_phone_level_hmms_transition_matrices(sentence_hmm, lexicon)
