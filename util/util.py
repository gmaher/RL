
def episode(act, update_state, reward, init_state, stop_function=None,
 update_agent=None, max_iter=1000, collect_history=False):
	'''
	Function to simulate an episode of a task for a particular 
	agent environment and reward

	inputs
	- act: function that takes in a state and returns an action index
	- update_state: function that takes in a state and action and returns
	a new state
	- reward: a function that takes in an action and a state argument
	and outputs a reward value
	-init_state: initial state from which to start the simulation,
	act(init_state) must be a valid function call
	-stop_function: function that takes in a state and returns true or false
	e.g. return true for a terminal state
	-update_agent: a function which takes in an action and reward value and a state
	and updates the agent
	-collect_history: boolean on whether to keep track of the states, actions and
	rewards
	'''

	it = 0

	actions = []
	states = []
	rewards = []

	s = init_state

	while it < max_iter:

		a = act(s)
		sprime = update_state(s,a)
		r = reward(a, s, sprime)

		if update_agent != None:
			update_agent(a,r,s)

		if collect_history:
			actions.append(a)
			rewards.append(r)
			states.append(sprime)

		if stop_function != None:
			if stop_function(sprime):
				break

		it = it+1

	return (states,actions,rewards)