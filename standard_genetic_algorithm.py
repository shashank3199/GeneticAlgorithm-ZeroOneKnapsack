"""
  standard_genetic_algorithm.py :   This file contains the standard implementation of Genetic Algorithm.
  File created by               :   Shashank Goyal
  Last commit done by           :   Shashank Goyal
  Last commit date              :   30th October 2020
"""

# these methods are used to select one or more objects from a list of items
from random import choice, choices

# numpy module for using genetic operations
import numpy as np
# tqdm module for progress bar
from tqdm import tqdm

# import knapsack object
from generate_data import Knapsack


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


class GeneticAlgorithm:
    """Class to implement Genetic Algorithm.

    Class Attributes:
        SINGLE_POINT_CROSSOVER (int)    : used to specify single point crossover scheme.
        TWO_POINT_CROSSOVER (int)       : used to specify two point crossover scheme.
        UNIFORM_CROSSOVER (int)         : used to specify uniform crossover scheme.

        BIT_FLIP_MUTATION (int)         : used to specify bit flip mutation scheme.
        SWAP_MUTATION (int)             : used to specify swap mutation scheme.

    Instance Attributes:
        cycle (int)                     : number of cycles, the GA is implemented.
        genome_size (int)               : size of the genome or chromosome.
        init_pop_size (int)             : initial size of the population.
        crossover_scheme (int)          : variable indicating the crossover scheme.
        mutation_scheme (int)           : variable indicating the mutation scheme.
        fitness_func (Callable)         : function that calculates the fitness.
        seed_range (Tuple[int, int])    : range in which the genome values can lie.
        encode (Callable)               : encode the genome sequence to genome value.
        decode (Callable)               : decode the genome value to genome sequence.

    """
    SINGLE_POINT_CROSSOVER = 1
    TWO_POINT_CROSSOVER = 2
    UNIFORM_CROSSOVER = 3

    BIT_FLIP_MUTATION = 1
    SWAP_MUTATION = 2

    def __init__(self, *args, **kwargs):
        """Initialization for Class.

        Args:
            *args       : Variable length argument list.
            **kwargs    : Arbitrary keyword arguments.

        """
        # set kwargs as instance attributes.
        self.__dict__.update(kwargs)
        # if range is not specified
        if self.seed_range is None:
            # set range from 0 to 2^n-1
            self.seed_range = (0, 2 ** self.genome_size - 1)
        # check that length of range is greater than initial population size
        assert (self.seed_range[1] - self.seed_range[0]) >= self.init_pop_size

    def init_population(self):
        """Initializes a population with unique genomes.

        Returns:
            np.ndarray  : Array containing genome values of individuals of the population.
        """
        # unpacking seed range
        low_range, high_range = self.seed_range
        # initialize population with no genomes
        population = np.array([], dtype=int)
        # loop until the length of population is less than self.init_pop_size
        while len(population) < self.init_pop_size:
            """
            1. Generate a random integer array.
            2. Remove repetitions.
            3. Append to existing population.
            4. Remove repetitions.
            """
            population = np.unique(np.append(population, np.unique(
                np.random.randint(size=self.init_pop_size - len(population), low=low_range, high=high_range))))
        # return population
        return population

    def selection(self, population: np.ndarray, selection_rate: float = 0.5):
        """Select the top percentage of population based on the fitness score. 

        Args:
            population      : input population.
            selection_rate  : percentage of population allowed to move to next step.

        Returns:
            nd.ndarray  : top `selection_rate` of the total population with high fitness scores.
        """
        # convert to list
        population = population.tolist()
        # sort in descending order based on fitness score
        population.sort(key=lambda p: self.fitness_func(p), reverse=True)
        # return the top `selection_rate` of population genomes
        return np.array(population[0:int(selection_rate * len(population))])

    def crossover(self, population: np.ndarray):
        """Generate the new generation after crossover.

        Args:
            population  : input population.

        Returns:
            np.ndarray  : initial population along with the new generation.
        """
        # save copy of initial population
        init_pop = population.copy()

        """
        Splitting the population in two partitions.
        """
        # initialize one population as empty
        pop_1 = np.array([], dtype=int)
        # generate set of indexes marking each individual in a population
        pop_id_set = list(range(len(population)))
        # loop through half population size
        for _ in range(len(population) // 2):
            # choose a random index
            chosen = choice(pop_id_set)
            # put the chosen indexed individual in pop_1
            pop_1 = np.append(pop_1, population[chosen].copy())
            # remove it from the set of options
            pop_id_set.remove(chosen)
        # put the rest of individuals in pop_2
        pop_2 = population[pop_id_set].copy()

        # iterate through pairs in population
        for i in range(len(population) // 2):
            # genome sequence of ith individual in pop_1
            p1 = self.decode(pop_1[i])
            # genome sequence of ith individual in pop_2
            p2 = self.decode(pop_2[i])

            # single point crossover
            if self.crossover_scheme == self.SINGLE_POINT_CROSSOVER:
                # choose an index
                index = choice(range(self.genome_size))
                # swap values from 0 to the chosen index
                pop_temp = p1[0:index].copy()
                p1[0:index] = p2[0:index].copy()
                p2[0:index] = pop_temp.copy()

            # two point crossover    
            elif self.crossover_scheme == self.TWO_POINT_CROSSOVER:
                # choose two indices
                chosen_idx = choices(list(range(self.genome_size)), k=2)
                # min becomes index 1
                index_1 = min(chosen_idx)
                # max becomes index 2
                index_2 = max(chosen_idx)

                # swap values from index 1 to index 2
                pop_temp = p1[index_1:index_2].copy()
                p1[index_1:index_2] = p2[index_1:index_2].copy()
                p2[index_1:index_2] = pop_temp.copy()

            # uniform crossover    
            elif self.crossover_scheme == self.UNIFORM_CROSSOVER:
                # generate an array of size `n` with values between 0 and 1
                uniform_seed = np.random.rand(self.genome_size)
                # iterate through the indices of the seed
                for j, u in enumerate(uniform_seed):
                    # if seed value at the index is greater than or equal to 50%
                    if u >= 0.5:
                        # swap the value at this index between the parents
                        t = p1[j].copy()
                        p1[j] = p2[j].copy()
                        p2[j] = t.copy()

            # encode the genome sequence back to value for pop_1
            pop_1[i] = self.encode(p1)
            # encode the genome sequence back to value for pop_2
            pop_2[i] = self.encode(p2)

        # append the pop_1 to init_pop
        init_pop = np.append(init_pop, pop_1)
        # append the pop_2 to init_pop
        init_pop = np.append(init_pop, pop_2)
        # return the population with 1st half as initial population and rest as new population
        return init_pop

    def mutation(self, population: np.ndarray):
        """Performs mutation at one or two positions in the genome sequence based on the scheme.

        Returns:
            np.ndarray  : initial population along with the mutated generation.
        """
        # iterate through the new generation only
        for i in range(len(population)//2-1, len(population)):
            # if the mutation rate is less than 50 %
            if np.random.rand(1)[0] < 0.5:
                # do nothing and move to next individual
                continue

            # genome sequence of ith individual in the new generation
            p = self.decode(population[i])

            # bit flip mutation
            if self.mutation_scheme == self.BIT_FLIP_MUTATION:
                # choose an index
                index = choice(range(self.genome_size))
                # flit the bit as the index
                p[index] = not (p[index])

            # swap mutation    
            elif self.mutation_scheme == self.SWAP_MUTATION:
                # choose two indices
                chosen_idx = choices(list(range(self.genome_size)), k=2)
                # min becomes index 1
                index_1 = min(chosen_idx)
                # max becomes index 2
                index_2 = max(chosen_idx)

                # swap the bits at the two indices
                t = p[index_1].copy()
                p[index_1] = p[index_2].copy()
                p[index_2] = t.copy()

            # encode the genome sequence back to value
            population[i] = self.encode(p)

        # return the population after removing the repetitions
        return np.unique(population)

    def driver(self, *args, **kwargs):
        """The driver method for the Genetic Algorithm.

        Args:
            *args       : Variable length argument list.
            **kwargs    : Arbitrary keyword arguments.            

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
        'cycle': 10,
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
    ga = GeneticAlgorithm(**genetic_algo_data)
    # run the driver method
    winner_genome = ga.driver()
    # print the results
    print("Sequence: {}\nGenome Value: {}\nProfit: {}\nCapacity Used: {}".format
          (get_genome_sequence(winner_genome, knapsack_object.n),
           winner_genome,
           fitness_func(winner_genome, knapsack_object),
           np.dot(get_genome_sequence(winner_genome, knapsack_object.n), knapsack_object.weights)))
