"""
  modified_genetic_algorithm.py :   This file contains the modified implementation of Genetic Algorithm.
  File created by               :   Shashank Goyal
  Last commit done by           :   Shashank Goyal
  Last commit date              :   30th October 2020
"""

# ThreadPoolExecutor and as_completed to run routines in parallel and get the result when it completes
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
Note: 	Here, the ThreadPoolExecutor is used and not ProcessPoolExecutor because 
        we can not use any objects that is not picklable and using the later
        results in pickling error.
"""

# numpy module for using genetic operations
import numpy as np
# tqdm module for progress bar
from tqdm import tqdm

# import knapsack object
from generate_data import Knapsack
# import the respective implementations of Genetic Algorithms
from standard_genetic_algorithm import GeneticAlgorithm
from island_genetic_algorithm import IslandGeneticAlgorithm
from restart_base_genetic_algorithm import RestartBaseGeneticAlgorithm


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


class ModifiedGeneticAlgorithm:
    """Class to implement RestartBase Genetic Algorithm.

    Attributes:
    cycle (int)						: number of cycles, the outer GA is implemented.
    m_parallel (int)				: number of parallel sub genetic algorithm routines.
    genome_size (int)				: size of the genome or chromosome.
    inner_genetic_algo_data (Dict)	: dictionary containing parameters for GeneticAlgorithm,
                                      See Docstring for `GeneticAlgorithm`.

    """

    def __init__(self, **kwargs):
        """Initialization for Class.

        Args:
            **kwargs    : Arbitrary keyword arguments.

        """
        # set kwargs as instance attributes.
        self.__dict__.update(kwargs)
        # create an empty list for sub genetic algorithm routines
        self.GA_list = []
        # calculate interval size for search space
        interval_size = (2 ** self.genome_size - 1) // self.m_parallel
        # iterate through the m_parallel sub routines
        for m in range(self.m_parallel):
            # copy dictionary
            m_inner_genetic_algo_data = self.inner_genetic_algo_data.copy()
            # update the range parameter
            m_inner_genetic_algo_data.update(seed_range=(m * interval_size, (m + 1) * interval_size))
            # append the genetic algorithm to list of sub routines
            self.GA_list.append(IslandGeneticAlgorithm(**m_inner_genetic_algo_data))

    def generate_superior_population(self):
        """Generate the superior population for a cycle

        Returns:
            np.ndarray : population containing local optimums for each interval
        """
        # iterate through the cycles
        for i in range(self.cycle):
            # parallelizing using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                # add the driver method along with arguments from GA_list to future instances
                futures = [executor.submit(GA.driver, 0.05, 10) for GA in self.GA_list]
                # initialize superior_population as empty array
                superior_population = np.array([], dtype=int)
                # as each sub routine is completed
                for f in as_completed(futures):
                    # stack horizontally over the population
                    superior_population = np.hstack((superior_population, f.result()))
                # remove repetitions
                superior_population = np.unique(superior_population)
            # yield this superior_population
            yield superior_population
            # print Cycle Completed message
            print(" Completed Cycle: {}".format(i + 1))

    def driver(self, threshold_vector: np.ndarray, threshold_value: float, restart_rate: float = 0.995):
        """The driver method for the RestartBase Genetic Algorithm.

        Args:
            threshold_vector	: vector could be weight or value vector
            threshold_value		: could be the minimum capacity or minimum profit
            restart_rate		: the threshold over which append a random population

        Returns:
            int : genome value of the winner genome
        """
        # create an RestartBaseGeneticAlgorithm object
        rga = RestartBaseGeneticAlgorithm(**self.inner_genetic_algo_data)
        # empty list for all the winners throughout the cycles
        winner_genomes = []
        # iterate through the cycles
        for s_pop in tqdm(self.generate_superior_population(), leave=False):
            # copy initial population from superior_population
            population = s_pop.copy()
            # loop until only one element is left in the population
            while len(population) > 1:
                # select the top 50%
                population = rga.selection(population)
                # perform crossover
                population = rga.crossover(population)
                # perform mutation
                population = rga.mutation(population)
                """
                generate a random value and if that is greater than restart threshold
                or
                only one individual in the population not satisfying the threshold conditions
                """
                if np.random.rand(1)[0] >= restart_rate or \
                        (len(population) == 1 and
                         np.dot(rga.decode(population[0]), threshold_vector) < threshold_value):
                    # append new random population
                    population = np.append(population, rga.init_population())
                    # perform selection of top 25%
                    population = rga.selection(population, 0.25)
            # add the winner genome of this cycle to the list
            winner_genomes.append(population[0])
        # choose the winner based on the maximum fitness scores out of the various winners
        best_genome = max(winner_genomes, key=lambda g: rga.fitness_func(g))
        # return the winner value
        return best_genome


if __name__ == "__main__":
    # name of file to load contents from
    fname = "./values/15_values.json"
    # load the knapsack object from the file
    knapsack_object = Knapsack(15, json_fname=fname)
    # convert knapsack vectors to numpy arrays
    knapsack_object.to_numpy()
    # number of sub parallel genetic algorithms
    m_parallel_threads = 10
    # values for the inner genetic algorithm instance
    inner_genetic_algo_data = {
        'cycle': 5,
        'genome_size': knapsack_object.n,
        'init_pop_size': (knapsack_object.n ** 2) // m_parallel_threads,
        'crossover_scheme': GeneticAlgorithm.UNIFORM_CROSSOVER,
        'mutation_scheme': GeneticAlgorithm.BIT_FLIP_MUTATION,
        'fitness_func': lambda genome: fitness_func(genome, knapsack_object),
        'seed_range': (0, 2 ** knapsack_object.n - 1),
        'encode': get_genome_value,
        'decode': lambda genome: get_genome_sequence(genome, knapsack_object.n)
    }

    # values for the outer genetic algorithm instance
    outer_genetic_algo_data = {
        'cycle': 5,
        'm_parallel': m_parallel_threads,
        'genome_size': knapsack_object.n,
        'inner_genetic_algo_data': inner_genetic_algo_data
    }

    # create an object
    mga = ModifiedGeneticAlgorithm(**outer_genetic_algo_data)
    # run the driver method
    winner_genome = mga.driver(knapsack_object.weights, 4000)
    # print the results
    print("Sequence: {}\nGenome Value: {}\nProfit: {}\nCapacity Used: {}".format
          (get_genome_sequence(winner_genome, knapsack_object.n),
           winner_genome,
           fitness_func(winner_genome, knapsack_object),
           np.dot(get_genome_sequence(winner_genome, knapsack_object.n), knapsack_object.weights)))
