from agent_base import AgentBase
import numpy as np

class Greedy(AgentBase):
	'''
	This class implements a greedy agent. The greedy
	agent acts by maintaining estimates of the average
	value of each action and always selects the highest
	one
	'''
	def __init__(self, initial_values):
		AgentBase.__init__(self)

		num_arms = len(initial_values)
		self.num_arms = num_arms
		self.value_estimates = initial_values
		self.num_observations = np.ones((num_arms))

	def act(self):
		return np.argmax(self.value_estimates)

	def update(self,action,reward):
		self.num_observations[action] += 1
		qold = self.value_estimates[action]
		qnew = qold + 1.0/(self.num_observations[action])*(reward-qold)
		self.value_estimates[action] = qnew