"""
  standard_genetic_algorithm.py :   This file contains the standard implementation of Genetic Algorithm.
  File created by       		:   Shashank Goyal
  Last commit done by   		:   Shashank Goyal
  Last commit date      		:   18th October 2020
"""

import numpy as np
from random import choice, choices

from generate_data import Knapsack

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


class GeneticAlgorithm:

    SINGLE_POINT_CROSSOVER = 1
    TWO_POINT_CROSSOVER = 2
    UNIFORM_CROSSOVER = 3

    BIT_FLIP_MUTATION = 1
    SWAP_MUTATION = 2

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        if self.seed_range is None:
            self.seed_range = (0, 2**self.genome_size - 1)
        assert (self.seed_range[1] - self.seed_range[0]) >= self.init_pop_size

    def init_population(self):
        low_range, high_range = self.seed_range
        population = np.array([], dtype=int)
        while len(population) < self.init_pop_size:
            population = np.unique(np.append(population,
                                             np.unique(np.random.randint(
                                                 size=self.init_pop_size -
                                                 len(population),
                                                 low=low_range, high=high_range))))
        return population

    def selection(self, population, selection_rate=0.5):
        population = population.tolist()
        population.sort(key=lambda p: self.fitness_func(p), reverse=True)
        return np.array(population[0:int(selection_rate * len(population))])

    def crossover(self, population):

        init_pop = population.copy()
        pop_1 = np.array([], dtype=int)
        pop_id_set = list(range(len(population)))
        for i in range(len(population) // 2):
            chosen = choice(pop_id_set)
            pop_1 = np.append(pop_1, population[chosen].copy())
            pop_id_set.remove(chosen)
        pop_2 = population[pop_id_set].copy()

        for i in range(len(population) // 2):
            p1 = self.decode(pop_1[i])
            p2 = self.decode(pop_2[i])
            if self.crossover_scheme == self.SINGLE_POINT_CROSSOVER:
                index = choice(range(self.genome_size))
                pop_temp = p1[0:index].copy()
                p1[0:index] = p2[0:index].copy()
                p2[0:index] = pop_temp.copy()
            elif self.crossover_scheme == self.TWO_POINT_CROSSOVER:
                chosen_idx = choices(list(range(self.genome_size)), k=2)
                index_1 = min(chosen_idx)
                index_2 = max(chosen_idx)

                pop_temp = p1[index_1:index_2].copy()
                p1[index_1:index_2] = p2[index_1:index_2].copy()
                p2[index_1:index_2] = pop_temp.copy()
            elif self.crossover_scheme == self.UNIFORM_CROSSOVER:
                uniform_seed = np.random.rand(self.genome_size)
                for j, u in enumerate(uniform_seed):
                    if u >= 0.5:
                        t = p1[j].copy()
                        p1[j] = p2[j].copy()
                        p2[j] = t.copy()

            pop_1[i] = self.encode(p1)
            pop_2[i] = self.encode(p2)

        init_pop = np.append(init_pop, pop_1)
        init_pop = np.append(init_pop, pop_2)
        return np.unique(init_pop)

    def mutation(self, population):

        for i in range(len(population)):
            if np.random.rand(1)[0] < 0.5:
                continue

            p = self.decode(population[i])
            if self.mutation_scheme == self.BIT_FLIP_MUTATION:
                index = choice(range(self.genome_size))
                p[index] = not(p[index])
            elif self.mutation_scheme == self.SWAP_MUTATION:
                id_set = list(range(self.genome_size))
                index_1 = choice(id_set)
                id_set.remove(index_1)
                index_2 = choice(id_set)
                t = p[index_1].copy()
                p[index_1] = p[index_2].copy()
                p[index_2] = t.copy()

            population[i] = self.encode(p)
        return np.unique(population)

    def driver(self):
        winner_genomes = []
        for i in range(self.cycle):
            population = self.init_population()
            while len(population) > 1:
                population = self.selection(population)
                population = self.crossover(population)
                population = self.mutation(population)
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

    ga = GeneticAlgorithm(**genetic_algo_data)
    best_genome = ga.driver()
    print("Sequence:", get_genome_sequence(best_genome, n),
          "\nGenome Value:", best_genome,
          "\nProfit:", fitness_func(best_genome),
          "\nCapacity Used:", np.dot(get_genome_sequence(best_genome, n), knapsack_object.weights))
