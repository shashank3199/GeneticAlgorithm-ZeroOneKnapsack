"""
  restart_base_genetic_algorithm.py :   This file contains the restart-base implementation of Genetic Algorithm.
  File created by                   :   Shashank Goyal
  Last commit done by               :   Shashank Goyal
  Last commit date                  :   26th October 2020
"""

import numpy as np
from standard_genetic_algorithm import GeneticAlgorithm
from generate_data import Knapsack

if __name__ == "__main__":
    fname = "./values/15_values.json"
    knapsack_object = Knapsack(15, json_fname=fname)
    knapsack_object.toNumpy()


def get_genome_sequence(code, padding):
    return np.array([int(x) for x in np.binary_repr(code, padding)])


def get_genome_value(genome):
    return int('0b' + ''.join([str(i) for i in genome.tolist()]), 2)


def fitness_func(genome_seed):
    global knapsack_object
    genome = get_genome_sequence(genome_seed, knapsack_object.n)
    if np.dot(genome, knapsack_object.weights) <= knapsack_object.capacity:
        return np.dot(genome, knapsack_object.values)
    return np.NINF


class RestartBaseGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, *args, **kwargs):
        super().__dict__.update(kwargs)

    def driver(self, threshold_vector, threshold_value, restart_rate=0.995):
        winner_genomes = []
        for i in range(self.cycle):
            population = self.init_population()
            while len(population) > 1:
                population = self.selection(population)
                population = self.crossover(population)
                population = self.mutation(population)
                restart_seed = np.random.rand(1)[0]
                if restart_seed >= restart_rate or (len(population) == 1 and np.dot(self.decode(population[0]), threshold_vector) < threshold_value):
                    population = np.append(population, self.init_population())
                    population = self.selection(population)
            winner_genomes.append(population[0])
        best_genome = max(winner_genomes, key=lambda g: self.fitness_func(g))
        return (best_genome)

if __name__ == "__main__":

    n = knapsack_object.n
    genetic_algo_data = {
        'cycle': 100,
        'genome_size': n,
        'init_pop_size': n**2,
        'crossover_scheme': GeneticAlgorithm.UNIFORM_CROSSOVER,
        'mutation_scheme': GeneticAlgorithm.BIT_FLIP_MUTATION,
        'fitness_func': fitness_func,
        'seed_range': (0, 2**n - 1),
        'encode': get_genome_value,
        'decode': lambda genome: get_genome_sequence(genome, n)
    }

    ga = RestartBaseGeneticAlgorithm(**genetic_algo_data)
    best_genome = ga.driver(knapsack_object.weights, int(knapsack_object.capacity * 0.75))
    # best_genome = ga.driver(knapsack_object.values, int(np.sum(knapsack_object.values) * 0.75))
    print("Sequence:", get_genome_sequence(best_genome, n),
          "\nGenome Value:", best_genome,
          "\nProfit:", fitness_func(best_genome),
          "\nCapacity Used:", np.dot(get_genome_sequence(best_genome, n), knapsack_object.weights))
