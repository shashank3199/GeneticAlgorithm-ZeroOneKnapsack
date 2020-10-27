"""
  modified_genetic_algorithm.py :   This file contains the modified implementation of Genetic Algorithm.
  File created by               :   Shashank Goyal
  Last commit done by           :   Shashank Goyal
  Last commit date              :   27th October 2020
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from standard_genetic_algorithm import GeneticAlgorithm
from restart_base_genetic_algorithm import RestartBaseGeneticAlgorithm
from island_genetic_algorithm import IslandGeneticAlgorithm
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


class ModifiedGeneticAlgorithm():

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.GA_list = []
        # interval_size = (2**self.genome_size-1)//self.m_parallel
        for m in range(self.m_parallel):
            inner_genetic_algo_data = self.inner_genetic_algo_data.copy()
            inner_genetic_algo_data.update(seed_range=(2**m, 2**(m+1)))
            # inner_genetic_algo_data.update(seed_range=(m*interval_size, (m+1)*interval_size))
            self.GA_list.append(IslandGeneticAlgorithm(**self.inner_genetic_algo_data))

    def generate_superior_population(self):
        for i in range(self.cycle):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(GA.driver, 0.05, 10) for GA in self.GA_list]
                superior_population = np.array([], dtype=int)
                for f in as_completed(futures):
                    superior_population = np.hstack((superior_population, f.result()));
                superior_population = np.unique(superior_population)
            yield superior_population

    def driver(self, threshold_vector, threshold_value, restart_rate=0.995):
        RGA = RestartBaseGeneticAlgorithm(**self.inner_genetic_algo_data)
        winner_genomes = []
        for s_pop in self.generate_superior_population():
            population = s_pop.copy()
            while len(population) > 1:
                population = RGA.selection(population)
                population = RGA.crossover(population)
                population = RGA.mutation(population)
                if np.random.rand(1)[0] >= restart_rate or (len(population) == 1 and np.dot(RGA.decode(population[0]), threshold_vector) < threshold_value):
                    population = np.append(population, RGA.init_population())
                    population = RGA.selection(population, 0.25)
            winner_genomes.append(population[0])
        best_genome = max(winner_genomes, key=lambda g: RGA.fitness_func(g))
        return (best_genome)
    

if __name__ == "__main__":

    n = knapsack_object.n
    inner_genetic_algo_data = {
        'cycle': 5,
        'genome_size': n,
        'init_pop_size': n**2,
        'crossover_scheme': GeneticAlgorithm.UNIFORM_CROSSOVER,
        'mutation_scheme': GeneticAlgorithm.BIT_FLIP_MUTATION,
        'fitness_func': fitness_func,
        'seed_range': (0, 2**n - 1),
        'encode': get_genome_value,
        'decode': lambda genome: get_genome_sequence(genome, n)
    }

    outer_genetic_algo_data = {
        'cycle': 5,
        'm_parallel': 20,
        'genome_size': n,
        'inner_genetic_algo_data': inner_genetic_algo_data
    } 

    mga = ModifiedGeneticAlgorithm(**outer_genetic_algo_data)
    best_genome = mga.driver(knapsack_object.weights, int(knapsack_object.capacity * 0.75))
    # best_genome = mga.driver(knapsack_object.values, int(np.sum(knapsack_object.values) * 0.75))
    print("Sequence:", get_genome_sequence(best_genome, n),
          "\nGenome Value:", best_genome,
          "\nProfit:", fitness_func(best_genome),
          "\nCapacity Used:", np.dot(get_genome_sequence(best_genome, n), knapsack_object.weights))
    