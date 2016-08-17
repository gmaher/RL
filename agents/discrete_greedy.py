from agent_base import AgentBase
import numpy as np

class DiscreteGreedy(AgentBase):
	'''
	This class implements a greedy agent for discrete environments, with
	discrete actions. The greedy
	agent acts by maintaining estimates of the average
	value of each action and always selects the action with the
	maximum action-value with probability 1-eps and with probability
	eps a random action is selected
	'''
	def __init__(self, initial_values, eps=0):
		'''
		inputs
		-initial_values: 2d array, array with initial values to use
		for action-value estimates. Dimensions must be number of states
		x number of actions
		-eps: float, probability with which a random action will
		be selected
		'''
		AgentBase.__init__(self)

		self.num_states, self.num_actions = initial_values.shape
		self.value_estimates = initial_values
		self.num_observations = np.ones_like(initial_values)
		self.eps = eps

	def act(self, s=0):
		x = np.random.rand()
		if x >= self.eps:
			return np.argmax(self.value_estimates[s,:])
		else:
			return np.random.randint(0,self.num_actions)

	def update(self,action,reward, state):
		self.num_observations[state,action] += 1
		qold = self.value_estimates[state,action]
		qnew = qold + 1.0/(self.num_observations[state,action])*(reward-qold)
		self.value_estimates[state,action] = qnew