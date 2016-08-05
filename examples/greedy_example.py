import sys
import os
sys.path.insert(0, os.path.abspath("../"))

from agents.greedy import Greedy
from environments.n_armed_bandit import N_ArmedBandit

import plotly as py
import plotly.graph_objs as go
import numpy as np

'''
N-armed bandit example

This example tests out the greedy and eps-greedy agents on an
N-armed bandit problem
'''

Niter = 100

means = [1,2,4,2,1]
stdevs = [1,1,1,1,1]

pessimistic_guess = [1,1,1,1,1]
optimistic_guess = [5,5,5,5,5]

pes = Greedy(initial_values=pessimistic_guess)
opt = Greedy(initial_values=optimistic_guess)

env = N_ArmedBandit(means=means,stdevs=stdevs)

opt_actions = []
opt_rewards = []
pes_actions = []
pes_rewards = []

T = range(0,Niter)

for t in T:

	a_opt = opt.act()
	a_pes = pes.act()

	r_opt = env.reward(a_opt)
	r_pes = env.reward(a_pes)

	opt.update(a_opt,r_opt)
	pes.update(a_pes,r_pes)

	opt_actions.append(a_opt)
	pes_actions.append(a_pes)

	opt_rewards.append(r_opt)
	pes_rewards.append(r_pes)

best = np.cumsum((np.ones(Niter)*means[2]))

trace0 = go.Scatter(
	x = T,
	y = np.cumsum(opt_rewards, dtype=np.float32)/best,
	mode = 'lines',
	name = 'optimistic agent'
	)

trace1 = go.Scatter(
	x = T,
	y = np.cumsum(pes_rewards, dtype=np.float32)/best,
	mode = 'lines',
	name = 'pessimistic agent'
	)

data = [trace0, trace1]

py.offline.plot(data, filename='plot.html')