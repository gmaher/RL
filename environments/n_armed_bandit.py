from environment_base import EnvironmentBase
import numpy as np

class N_ArmedBandit(EnvironmentBase):

	def __init__(self, means=[], stdevs=[]):

		EnvironmentBase.__init__(self)

		if len(means) != len(stdevs):
			raise ValueError("N_ArmedBandit, means and stdevs have different length")

		self.means = means
		self.stdevs = stdevs

	def reward(self, action):

		return stdevs[action]*np.random.randn()+means[action]