class AgentBase:

	def __init__(self):

	def act(self):
		'''
		The act function should return an action based on the
		agent's current status
		'''

		raise RuntimeError("act method not implemented for class "+\
			self.__class__)

	def fit(self):
		'''
		the fit function is for agents that should be updated
		with an already existing dataset

		e.g. we already have a list of (state,action,reward)
		tuples which we wish to use to initialize the agent
		'''

		raise RuntimeError("fit method not implemented for class "+\
		 self.__class__)

	def update(self):
		'''
		the update function is for agents that need the 
		ability to update themselves incrementely.

		e.g. a greedy method that receives one new
		observation and must update its action-value
		estimates
		'''

		raise RuntimeError("update method not implemented for class "+\
		 self.__class__)