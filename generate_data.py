"""
  generate_data.py      :   This file contains the Knapsack class for creating random weight-value vectors and writing it to json files.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   18th October 2020
"""

import numpy as np
from random import choice
import json

n = 25
seed = 51
fname = './values/' + str(n)+'_values.json'

class Knapsack:
	def __init__(self, n, upper_seed=51, json_fname=None):
		if json_fname is not None:
			with open(json_fname, 'r') as file:
				self.__dict__ = json.load(file)
			return

		self.n = n
		self.values = np.cumsum(np.random.randint(low=1, high=upper_seed, size=n))
		self.weights = (np.cumsum(np.multiply(self.values, np.random.rand(n,)))).astype(np.int)
		self.values += self.weights
		self.capacity= int(3 * choice(self.weights[int(-0.1*n):]))

		self.values = self.values.tolist()
		self.weights = self.weights.tolist()

	def __repr__(self):
		string = "Capacity: {}\n".format(self.capacity)
		string += "Number of Items: {}\n".format(self.n)
		string += "Weights: {}\n".format(repr(self.weights))
		string += "Values: {}\n".format(repr(self.values))
		return string

	def toNumpy(self):
		self.weights = np.array(self.weights, dtype=np.int)
		self.values = np.array(self.values, dtype=np.int)

if __name__ == "__main__":
	k = Knapsack(n, seed)
	print (k)

	with open(fname, 'w') as file:
	    json.dump(k.__dict__, file)

	k_load = Knapsack(n, json_fname=fname)
	print (k_load)
	print ("Sanity Check Complete")