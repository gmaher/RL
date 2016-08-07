import sys
import os
sys.path.insert(0, os.path.abspath("../"))

from agents.greedy import Greedy
from environments.n_armed_bandit import N_ArmedBandit
from util.util import episode

import plotly as py
import plotly.graph_objs as go
import numpy as np

'''
N-armed bandit example

This example tests out the greedy and eps-greedy agents on an
N-armed bandit problem
'''

Niter = 500
T = range(0,Niter)

means = [1,2,10,2,1]
stdevs = [3,3,3,3,3]

pessimistic_guess = [1,1,1,1,1]
optimistic_guess = [5,5,5,5,5]

pes = Greedy(initial_values=pessimistic_guess)
opt = Greedy(initial_values=optimistic_guess)
eps = Greedy(initial_values=pessimistic_guess, eps=0.05)

env = N_ArmedBandit(means=means,stdevs=stdevs)

opt_states, opt_actions, opt_rewards = \
episode(opt.act, env.update_state, env.reward, 1, update_agent=opt.update,
 max_iter=Niter, collect_history=True)

pes_states, pes_actions, pes_rewards = \
episode(pes.act, env.update_state, env.reward, 1, update_agent=pes.update,
 max_iter=Niter, collect_history=True)

eps_states, eps_actions, eps_rewards = \
episode(eps.act, env.update_state, env.reward, 1, update_agent=eps.update,
 max_iter=Niter, collect_history=True)

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

trace2 = go.Scatter(
	x = T,
	y = np.cumsum(eps_rewards, dtype=np.float32)/best,
	mode = 'lines',
	name = 'epsilon greedy'
	)

data = [trace0, trace1, trace2]

py.offline.plot(data, filename='plot.html')