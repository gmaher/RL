import sys
import os
sys.path.insert(0, os.path.abspath("../"))

from agents.discrete_greedy import DiscreteGreedy
from environments.grid_world import GridWorld
from util.util import episode

import plotly as py
import plotly.graph_objs as go
import numpy as np

'''
grid world example

'''
num_rows = 4
num_columns = 4
num_states = num_rows*num_columns

num_sim = 3000

env = GridWorld(rows=num_rows, columns=num_columns)

initial_values = np.zeros((num_states,4))
agent = DiscreteGreedy(initial_values, eps=0.1)

def reward(a,s,sprime):
	if (sprime==0) or (sprime == num_rows*num_columns-1):
		return 1
	else:
		return -1

def stop(sprime):
	if (sprime==0) or (sprime == num_rows*num_columns-1):
		return True
	else:
		return False

for i in range(0,num_sim):
	episode(agent.act, env.update_state, reward, 6,
		update_agent=agent.update, stop_function=stop)

print agent.value_estimates

print np.mean(agent.value_estimates,axis=1).reshape((4,4))