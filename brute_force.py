"""
  brute_force.py 		:   This file contains brute force code for the 0-1 knapsack problem.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   30th October 2020
"""

# numpy module for dot product and encoding scheme
import numpy as np
# tqdm module for progress bar
from tqdm import tqdm

# import knapsack object
from generate_data import Knapsack

if __name__ == "__main__":
    # name of file to load contents from
    fname = "./values/15_values.json"
    # display name of loaded file
    print("Loaded File: {}\n".format(fname))
    # load the knapsack object from the file
    knapsack_object = Knapsack(15, json_fname=fname)
    # convert knapsack vectors to numpy arrays
    knapsack_object.to_numpy()
    # create an list to store solutions
    solutions = []

    # iterate through 2^(n-1) possible combinations
    for i in tqdm(range(2 ** knapsack_object.n - 1)):
        try:
            # convert each value into binary representation
            genome = np.array([int(x) for x in np.binary_repr(i, knapsack_object.n)], dtype=np.byte)
            # calculate dot product of the genome sequence with weights vectors
            if np.dot(genome, knapsack_object.weights) <= knapsack_object.capacity:
                # if the load can be carried append to the solutions list as  -
                # (<Genome Sequence> , <Capacity Used>, <Total Profit>)
                solutions.append(
                    (genome, np.dot(genome, knapsack_object.weights), np.dot(genome, knapsack_object.values)))
        # exception in case of Keyboard Interruption by user
        except KeyboardInterrupt:
            # display the last calculated value
            print("Stopped at:", i)

    """
    # display all values sorted in descending order based on the Total Profit
    solutions = sorted(solutions, key=lambda s: s[2], reverse=True)
    for s in solutions:
        print("\nGenome: {}\nCapacity Used: {}\nTotal Profit: {}".format(*s))	
    """

    # calculate the highest profitable sequence based on the Total Profit
    solution = max(solutions, key=lambda s: s[2])
    # display the best possible combination
    print("\nGenome: {}\nCapacity Used: {}\nTotal Profit: {}".format(*solution))
