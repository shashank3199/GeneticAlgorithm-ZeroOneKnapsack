"""
  restart_base_genetic_algorithm.py :   This file contains the restart-base implementation of Genetic Algorithm.
  File created by                   :   Shashank Goyal
  Last commit done by               :   Shashank Goyal
  Last commit date                  :   30th October 2020
"""

# numpy module for using genetic operations
import numpy as np
# tqdm module for progress bar
from tqdm import tqdm

# import knapsack object
from generate_data import Knapsack
# import the GeneticAlgorithm class to be inherited
from standard_genetic_algorithm import GeneticAlgorithm


def get_genome_sequence(code: int, padding: int):
    """
    This method is used to convert a base-10 number to its
    base-2 representation as an array.

    Args:
        code (int)      : It is the base-10 number to be converted.
        padding (int)   : It is the length of the array representation.

    Examples:
        >>> get_genome_sequence(7,2)
        array([1, 1, 1])
        >>> get_genome_sequence(7,3)
        array([1, 1, 1])
        >>> get_genome_sequence(7,4)
        array([0, 1, 1, 1])
        >>> get_genome_sequence(7,5)
        array([0, 0, 1, 1, 1])

    Returns:
        np.ndarray  : Array containing 0s and 1s representing base-2 of the `code`.
    """
    return np.array([int(x) for x in np.binary_repr(code, padding)])


def get_genome_value(genome: np.ndarray):
    """
    This method converts a numpy array (of zeros and ones) from base-2
    to corresponding base-10 value.

    Args:
        genome (np.ndarray) : The array containing sequence of 0s and 1s.

    Examples:
        >>> get_genome_value(np.array([1,1,0]))
        6
        >>> get_genome_value(np.array([1,0,0]))
        4
        >>> get_genome_value(np.array([0,1,0,0]))
        4
        >>> get_genome_value(np.array([0,0,1,0,0]))
        4

    Returns:
        int : Base-10 value of the `genome` array.
    """
    return int('0b' + ''.join([str(i) for i in genome.tolist()]), 2)


def fitness_func(code: int, knapsack_obj: Knapsack):
    """
    This method calculates the profit that can be achieved for a specific genome
    and returns it as the fitness of the genome.

    Args:
        code (int)               : It is the base-10 genome value.
        knapsack_obj (Knapsack)  : It is the object containing the vectors for the problem.

    Returns:
        int : Total profit value, if the weight load can be taken else `np.NINF`.
    """
    # get genome sequence
    genome = get_genome_sequence(code, knapsack_obj.n)
    # check if total load of genome fits in capacity
    if np.dot(genome, knapsack_obj.weights) <= knapsack_obj.capacity:
        # return the profit
        return np.dot(genome, knapsack_obj.values)
    # return Negative Infinity 
    return np.NINF


class RestartBaseGeneticAlgorithm(GeneticAlgorithm):
    """Class to implement RestartBase Genetic Algorithm.

    Attributes:
        inherited from super() class
    """

    def __init__(self, *args, **kwargs):
        """Initialization for Class.

        Args:
            *args       : Variable length argument list.
            **kwargs    : Arbitrary keyword arguments.

        """
        # set kwargs as super() attributes.
        super().__init__(*args, **kwargs)
        # set kwargs as instance attributes.
        super().__dict__.update(kwargs)

    def driver(self, threshold_vector: np.ndarray, threshold_value: float, restart_rate: float = 0.995):
        """The driver method for the RestartBase Genetic Algorithm.

        Args:
            threshold_vector	: vector could be weight or value vector
            threshold_value		: could be the minimum capacity or minimum profit
            restart_rate		: the threshold over which append a random population

        Returns:
            int : genome value of the winner genome
        """
        # empty list for all the winners throughout the cycles
        winner_genomes = []
        # iterate through the cycles
        for _ in tqdm(range(self.cycle), leave=False):
            # create initial population
            population = self.init_population()
            # loop until only one element is left in the population
            while len(population) > 1:
                # select the top 50%
                population = self.selection(population)
                # perform crossover
                population = self.crossover(population)
                # perform mutation
                population = self.mutation(population)
                """
                generate a random value and if that is greater than restart threshold
                or
                only one individual in the population not satisfying the threshold conditions
                """
                if np.random.rand(1)[0] >= restart_rate or \
                        (len(population) == 1 and
                         np.dot(self.decode(population[0]), threshold_vector) < threshold_value):
                    # append new random population
                    population = np.append(population, self.init_population())
                    # perform selection of top 50%
                    population = self.selection(population)
            # add the winner genome of this cycle to the list        
            winner_genomes.append(population[0])
        # choose the winner based on the maximum fitness scores out of the various winners        
        best_genome = max(winner_genomes, key=lambda g: self.fitness_func(g))
        # return the winner value
        return best_genome


if __name__ == "__main__":
    # name of file to load contents from
    fname = "./values/15_values.json"
    # load the knapsack object from the file
    knapsack_object = Knapsack(15, json_fname=fname)
    # convert knapsack vectors to numpy arrays
    knapsack_object.to_numpy()
    # values for the genetic algorithm instance
    genetic_algo_data = {
        'cycle': 5,
        'genome_size': knapsack_object.n,
        'init_pop_size': knapsack_object.n ** 2,
        'crossover_scheme': GeneticAlgorithm.UNIFORM_CROSSOVER,
        'mutation_scheme': GeneticAlgorithm.BIT_FLIP_MUTATION,
        'fitness_func': lambda genome: fitness_func(genome, knapsack_object),
        'seed_range': (0, 2 ** knapsack_object.n - 1),
        'encode': get_genome_value,
        'decode': lambda genome: get_genome_sequence(genome, knapsack_object.n)
    }

    # create an object
    ga = RestartBaseGeneticAlgorithm(**genetic_algo_data)
    # run the driver method
    winner_genome = ga.driver(knapsack_object.weights, int(knapsack_object.capacity * 0.75))
    # print the results
    print("Sequence: {}\nGenome Value: {}\nProfit: {}\nCapacity Used: {}".format
          (get_genome_sequence(winner_genome, knapsack_object.n),
           winner_genome,
           fitness_func(winner_genome, knapsack_object),
           np.dot(get_genome_sequence(winner_genome, knapsack_object.n), knapsack_object.weights)))
