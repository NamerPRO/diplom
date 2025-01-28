import time
from collections import defaultdict
from typing import Optional, Final, List, Dict

from pyfoma import State

from core.transducers.grammar.language_models import KatzSmoothingLM
from utils.ngram import NGram
from utils.wfst import WFST


class LexiconTransducer:
    SILENCE: Final[str] = "<sil>"
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
        wfst = WFST()
        loop_state = wfst.add_state()
        wfst.mark_as_final(loop_state)
        wfst.add_arc(wfst.start_state, loop_state, (WFST.EPS, WFST.EPS))
        silence_state = wfst.add_state()
        wfst.add_arc(wfst.start_state, silence_state, (LexiconTransducer.SILENCE, WFST.EPS))
        homophones: Dict[str, List] = defaultdict(lambda: [0, list()])
        with open(self.__path_to_lexicon_file, "r") as f:
            for line in f:
                word, phones_str = line.split(" ", 1)
                phones_str = phones_str.rstrip('\n')
                phones = phones_str.split(" ")
                prev_state = loop_state
                for i, phone in enumerate(phones):
                    cur_state = wfst.add_state()
                    wfst.add_arc(prev_state, cur_state, (phone, WFST.EPS if i > 0 else word))
                    prev_state = cur_state
                homophones[phones_str][1].append(prev_state)
                homophones[phones_str][0] += 1
        unknown_state = wfst.add_state()
        unknown_state_token = NGram.get_sys_token('unknown')
        homophones[unknown_state_token][1].append(unknown_state)
        homophones[unknown_state_token][0] += 1
        wfst.add_arc(loop_state, unknown_state, (unknown_state_token, unknown_state_token))
        silence_disambiguation_number = self.__add_disambiguation_symbols(wfst, homophones, loop_state, silence_state)
        silence_disambiguation_symbol = f"#{silence_disambiguation_number}"
        wfst.add_arc(silence_state, loop_state, (silence_disambiguation_symbol, WFST.EPS))
        wfst.add_arc(loop_state, loop_state, (KatzSmoothingLM.BACKOFF_DISAMBIGUITY_SYMBOL, KatzSmoothingLM.BACKOFF_DISAMBIGUITY_SYMBOL))
        return wfst

    def __add_disambiguation_symbols(self, wfst: WFST, homophones: Dict[str, List], loop_state: int,
                                     silence_state: int) -> int:
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
                wfst.add_arc(state, loop_state, (disambiguation_label, WFST.EPS))
                next_state = wfst.add_state()
                wfst.add_arc(state, next_state, (disambiguation_label, WFST.EPS))
                wfst.add_arc(next_state, silence_state, (LexiconTransducer.SILENCE, WFST.EPS))
        return silence_disambiguation_number


if __name__ == "__main__":
    lexicon = LexiconTransducer(
        path_to_lexicon_file="C:/Users/PeterA/Desktop/vkr/test/lexicon.txt"
    )
    grammar = KatzSmoothingLM.from_train_corpus(
        n=2,
        corpus_path='C:/Users/PeterA/Desktop/vkr/test/corpus.txt',
        k=5
    )

    wfst_l = lexicon.wfst
    wfst_g = grammar.build_wfst()

    wfst_l.view()
    time.sleep(1)
    wfst_g.view()

    time.sleep(1)
    wfst_lg = wfst_l.compose(wfst_g)
    wfst_lg.view()

    time.sleep(1)
    wfst_lg.determinize()
    wfst_lg.minimize()
    wfst_lg.view()
