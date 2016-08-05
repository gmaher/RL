class EnvironmentBase:

	def __init__(self):
		'''
		Initializer, empty for now
		'''
	def reward(self):
		'''
		reward should take in an action and return a number based on
		the current state and the action taken
		'''

		raise RuntimeError("reward not implemented by "+\
			self.__class__)


	def update_state(self):
		'''
		update_state should take in an action and update the state
		to the new state and return it so that it can be accessed
		outside the local scope
		'''

		raise RuntimeError("update_state no implemented by "+\
			self.__class__)


	def reward_and_update(self):
		'''
		reward_and_update should return both the reward and new state
		after an action has been taken by the agent
		'''

		raise RuntimeError("reward_and_update not implemented by "+\
			self.__class__)