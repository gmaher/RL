from environment_base import EnvironmentBase
import numpy as np

class N_ArmedBandit(EnvironmentBase):

	def __init__(self, num_arms, means=None, stdevs=None):

		EnvironmentBase.__init__(self)

		self.num_arms = num_arms
		self.means = []
		self.stdevs = []

		if (means==None) or (stdevs==None):
			#generate standard normal action values
			

	def reward(self, action):

