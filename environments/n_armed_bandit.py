from environment_base import EnvironmentBase
import numpy as np

class N_ArmedBandit(EnvironmentBase):
	'''
	This class implements an N-armed bandit environment.
	There are N possible choices and each choice returns 
	a reward sampled from a normal distribution with mean
	means[action] and standard deviation stdves[action]
	'''
	def __init__(self, means=[], stdevs=[]):
		EnvironmentBase.__init__(self)

		if len(means) != len(stdevs):
			raise ValueError("N_ArmedBandit, means and stdevs have different length")

		self.means = means
		self.stdevs = stdevs

	def reward(self, action, state, state_prime):
		return self.stdevs[action]*np.random.randn()+self.means[action]

	def update_state(self, state, action):
		return 1