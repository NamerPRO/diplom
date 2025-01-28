from utils.wfst import WFST
import time


wfst = WFST("hello", render_state_names_str=True)
s0 = wfst.start_state
s1 = wfst.add_state('1')
s2 = wfst.add_state('2')
s3 = wfst.add_state('3')
s4 = wfst.add_state('4')
s5 = wfst.add_state('5')
s6 = wfst.add_state('6')
s7 = wfst.add_state('7')
s8 = wfst.add_state('8')
s9 = wfst.add_state('world')
s10 = wfst.add_state('10')
s11 = wfst.add_state('here')
wfst.add_arc('hello', '1', ('t', 'THIS'))
wfst.add_arc('hello', '5', ('i', "IS"))
wfst.add_arc('hello', '7', ('a', 'A'))
wfst.add_arc('hello', '8', ('t', 'TEST'))
wfst.add_arc('1', '2', ('h', WFST.EPS))
wfst.add_arc('2', '3', ('i', WFST.EPS))
wfst.add_arc('3', '4', ('s', WFST.EPS))
wfst.add_arc('5', '6', ('s', WFST.EPS))
wfst.add_arc('8', 'world', ('e', WFST.EPS))
wfst.add_arc('world', '10', ('s', WFST.EPS))
wfst.add_arc('10', 'here', ('t', WFST.EPS), 5.154)
wfst.mark_as_final('7')
wfst.mark_as_final('6')
wfst.mark_as_final('4')
wfst.mark_as_final('here', 2.435)
wfst.view()
time.sleep(1)
wfst.determinize()
wfst.view()
exit(0)


from pyfoma import FST, State
from utils.pyfoma_wfst import WFST


# wfst = FST.re("a<1.0> b<3.0>* c<5.0> | a<2.0> b<3.0>* d<6.0>") # Will be pseudo-determinized
# FST.render(wfst)
# time.sleep(1)
# wfstdet = wfst.determinize()
# FST.render(wfst)
# exit(0)

wfst = WFST("0")
s0 = wfst.start_state
s1 = wfst.add_state('1')
s2 = wfst.add_state('2')
s3 = wfst.add_state('3')
s4 = wfst.add_state('4')
s5 = wfst.add_state('5')
s6 = wfst.add_state('6')
s7 = wfst.add_state('7')
s8 = wfst.add_state('8')
s9 = wfst.add_state('9')
s10 = wfst.add_state('10')
s11 = wfst.add_state('11')
wfst.add_arc('0', '1', ('t', 'THIS'))
wfst.add_arc('0', '5', ('i', "IS"))
wfst.add_arc('0', '7', ('a', 'A'))
wfst.add_arc('0', '8', ('t', 'TEST'))
wfst.add_arc('1', '2', ('h', WFST.TRUE_EPS))
wfst.add_arc('2', '3', ('i', WFST.TRUE_EPS))
wfst.add_arc('3', '4', ('s', WFST.TRUE_EPS))
wfst.add_arc('5', '6', ('s', WFST.TRUE_EPS))
wfst.add_arc('8', '9', ('e', WFST.TRUE_EPS))
wfst.add_arc('9', '10', ('s', WFST.TRUE_EPS))
wfst.add_arc('10', '11', ('t', WFST.TRUE_EPS))
wfst.mark_as_final(s7.name)
wfst.mark_as_final(s6.name)
wfst.mark_as_final(s4.name)
wfst.mark_as_final(s11.name)
wfst.view()
time.sleep(1)
# wfst.minimize()
wfst.determinize()
wfst.view()












exit(0)

# # Create a non-deterministic transducer
# fst = WFST('q0')
#
# s1 = fst.start_state
# s2 = fst.add_state(name="q1")
# s3 = fst.add_state(name="q2")
#
# fst.add_arc("q0", "q1", ("a", "x"), 1)
# fst.add_arc("q0", "q1", ("a", "y"), 2)
# fst.add_arc("q1", "q2", ("b", "z"), 3)
#
# # Determinize the transducer
# fst.view()
# exit(0)

from utils.pyfoma_wfst import WFST

# wfst = WFST()

wfst = FST()

sx = State(name='ε')
wfst.initialstate = sx
s1 = State(name='b')
sx.add_transition(s1, ("a", "x"), 1)
s2, s3 = State(name='2'), State(name='3')
s1.add_transition(s2, ("b", "y"), 2)
s1.add_transition(s3, ("b", "z"), 3)
s4, s5 = State(name='4'), State(name='5')
s2.add_transition(s4, ("d", "w"), 4)
s3.add_transition(s5, ("d", "w"), 5)

wfst.states = {sx,s1,s2,s3,s4,s5}
wfst.alphabet = {"a","b","c","d"}
s4.finalweight = 0
s5.finalweight = 0
wfst.finalstates = {s4,s5}

FST.render(wfst)
# wfst.epsilon_remove().determinize().minimize()
# @ .compose

# noinspection PyUnresolvedReferences
# wfst.minimize()
wfst = wfst.determinize()

time.sleep(1)

FST.render(wfst)

time.sleep(1)

# wfst.push_weights()

# w = State(name='8')
# wfst.states.add(w)
# w.add_transition(sx, ("v", "v"), 1)
# wfst.determinize()
FST.render(wfst)

print(list(wfst.generate("abd", weights = True)))
# print(list(wfst.generate("d", weights = True)))


# myfst = FST()            # Init object
# s0 = myfst.initialstate  # FST() always has one state, make that s0
# s1 = State()             # Add a state
# s2 = State()
# s0.add_transition(s1, ("","x"), 1.0)  # Add transitions...
# s0.add_transition(s2, ("a","y"), 0.5)
# s1.add_transition(s0, ("a","a"), 0.0)
# s1.finalweight = 2.0                   # Set the final weight
# myfst.states = {s0,s1,s2}                 # Set of states
# myfst.finalstates = {s1}               # Set of final states
# myfst.alphabet = {"a","x","y"}         # Optional alphabet
#
# myfst.minimize()
# FST.render(myfst)
#
# print(list(myfst.generate("a", weights = True)))