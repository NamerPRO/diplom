import time

from pyfoma import FST, State

from utility.wfst import WFST

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
wfst.minimize()

time.sleep(1)

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