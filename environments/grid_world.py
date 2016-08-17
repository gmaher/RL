class GridWorld():

	def __init__(self, rows, columns):
		self.rows = rows
		self.columns = columns
		self.num_states = rows*columns

	def update_state(self, state, action):
		'''
		update state according grid world pattern

		inputs
		state - int, the current state, must be between
		0 and self.num_states-1
		action - int, the selected action, must be between 0 and 3
		0 = move up
		1 = move right
		2 = move down
		3 = move left
		'''
		current_column = state%self.rows
		current_row = state/self.rows

		if (action == 0) and (current_row != 0):
			return state-self.columns

		elif (action == 1) and (current_column != self.columns-1):
			return state+1

		elif (action == 2) and (current_row != self.rows-1):
			return state+self.columns

		elif (action == 3) and (current_column != 0):
			return state-1

		else:
			return state



