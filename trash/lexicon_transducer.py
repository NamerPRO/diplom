import time
from collections import defaultdict
from typing import Optional, Final, List, Dict

from pyfoma import State

from core.transducers.grammar.language_models import KatzSmoothingLM
from utils.ngram import NGram
from utils.pyfoma_wfst import WFST


class LexiconTransducer:
    SILENCE: Final[str] = "[sil]"
    DISAMBIGUATE_SYMBOL: Final[str] = '#1'

    def __init__(self, path_to_lexicon_file: str):
        self.__path_to_lexicon_file = path_to_lexicon_file
        self.__wfst: Optional[WFST] = None

    @property
    def wfst(self) -> WFST:
        if self.__wfst is None:
            self.__wfst = self.build_wfst()
        return self.__wfst

    def build_wfst(self):
        wfst = WFST(initial_state_name="INIT")
        loop_state = wfst.add_state(name="LOOP")
        wfst.mark_as_final(loop_state.name)
        wfst.add_arc(wfst.start_state.name, loop_state.name, (WFST.EPS, WFST.EPS))
        silence_state = wfst.add_state(name=LexiconTransducer.SILENCE)
        wfst.expand_alphabet(silence_state.name)
        wfst.add_arc(wfst.start_state.name, silence_state.name, (LexiconTransducer.SILENCE, WFST.EPS))
        homophones: Dict[str, List] = defaultdict(lambda: [0, list()])
        with open(self.__path_to_lexicon_file, "r") as f:
            for line in f:
                word, phones_str = line.split(" ", 1)
                phones_str = phones_str.rstrip('\n')
                phones = phones_str.split(" ")
                prev_state = loop_state
                for i, phone in enumerate(phones):
                    cur_state = wfst.add_state()
                    wfst.expand_alphabet(phone)
                    wfst.add_arc(prev_state.name, cur_state.name, (phone, WFST.EPS if i > 0 else word))
                    prev_state = cur_state
                homophones[phones_str][1].append(prev_state)
                homophones[phones_str][0] += 1
                wfst.expand_alphabet(phones[0])
        unknown_state = wfst.add_state(name=NGram.get_sys_token('unknown'))
        homophones[unknown_state.name][1].append(unknown_state)
        homophones[unknown_state.name][0] += 1
        wfst.add_arc(loop_state.name, unknown_state.name, (unknown_state.name, unknown_state.name))
        silence_disambiguation_number = self.__add_disambiguation_symbols(wfst, homophones, loop_state, silence_state)
        silence_disambiguation_symbol = f"#{silence_disambiguation_number}"
        wfst.add_arc(silence_state.name, loop_state.name, (silence_disambiguation_symbol, WFST.EPS))
        wfst.expand_alphabet(silence_disambiguation_symbol)
        wfst.add_arc(loop_state.name, loop_state.name, (KatzSmoothingLM.BACKOFF_DISAMBIGUITY_SYMBOL, KatzSmoothingLM.BACKOFF_DISAMBIGUITY_SYMBOL))
        return wfst

    def __add_disambiguation_symbols(self, wfst: WFST, homophones: Dict[str, List], loop_state: State,
                                     silence_state: State) -> int:
        silence_disambiguation_number = 1
        for homophone in homophones:
            count, states = homophones[homophone]
            for i, state in enumerate(states, start=1):
                if count > 0:
                    disambiguation_label = f"#{i}"
                    if silence_disambiguation_number == i:
                        silence_disambiguation_number += 1
                else:
                    disambiguation_label = LexiconTransducer.DISAMBIGUATE_SYMBOL
                wfst.add_arc(state.name, loop_state.name, (disambiguation_label, WFST.EPS))
                next_state = wfst.add_state()
                wfst.add_arc(state.name, next_state.name, (disambiguation_label, WFST.EPS))
                wfst.add_arc(next_state.name, silence_state.name, (LexiconTransducer.SILENCE, WFST.EPS))
                wfst.expand_alphabet(disambiguation_label)
        return silence_disambiguation_number


if __name__ == "__main__":
    lexicon = LexiconTransducer(
        path_to_lexicon_file="C:/Users/PeterA/Desktop/vkr/test/lexicon.txt"
    )
    grammar = KatzSmoothingLM.from_train_corpus(
        n=2,
        corpus_path='C:/Users/PeterA/Desktop/vkr/test/corpus.txt',
        k=5,
        reserved_probability=0.1
    )

    wfst_l = lexicon.wfst
    wfst_g = grammar.build_wfst()

    wfst_l.view()
    time.sleep(1)
    wfst_g.view()

    time.sleep(1)
    wfst_l.compose(wfst_g)
    wfst_l.view()

    time.sleep(1)
    wfst_l.determinize()
    wfst_l.minimize()
    wfst_l.view()
