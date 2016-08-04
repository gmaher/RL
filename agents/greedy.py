from agent_base import AgentBase
import numpy as np

class Greedy(AgentBase):

	def __init__(self, num_arms):

		AgentBase.__init__(self)

		self.num_arms = num_arms

		self.value_estimates = np.zeros((num_arms))

		self.num_observations = np.zeros((num_arms))

	def act():

		return np.argmax(self.value_estimates)

	def update(action,reward):

		self.num_observations[action] += 1

		qold = self.value_estimates[action]

		qnew = qold + 1.0/(self.num_observations[action])(reward-qold)

		self.value_estimates[action] = qnew