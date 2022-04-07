from copy import copy
from itertools import takewhile
from random import randint, random
import numpy as np


class Item:

    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

    def __repr__(self):
        return f'(weight={self.weight}, value={self.value})'


class Chromosome:

    MUTATION_PROBABILITY = 0.05

    def __init__(self, items, max_weight, gens=None):
        self.items = items
        self.max_weight = max_weight
        self.gens = gens if gens is not None else [randint(0, 1) for _ in range(len(items))]
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        taken_items = self.get_phenotype()
        taken_value = 0
        taken_weight = 0

        for item in taken_items:
            taken_value += item.value
            taken_weight += item.weight

        return taken_value if taken_weight <= self.max_weight else 0

    def get_phenotype(self):
        taken_items = []
        for gen, item in zip(self.gens, self.items):
            if gen:
                taken_items.append(item)
        return taken_items

    def mutate(self):
        for i in range(len(self.gens)):
            rand = random()
            if rand < self.MUTATION_PROBABILITY:
                self.gens[i] = 1 - self.gens[i]
        self.fitness = self.calc_fitness()

    def cross(self, other):
        cross_point = randint(0, len(self.gens))

        child1 = copy(self.gens)
        child2 = copy(other.gens)

        child1[cross_point:] = other.gens[cross_point:]
        child2[cross_point:] = self.gens[cross_point:]

        return Chromosome(self.items, self.max_weight, child1), Chromosome(self.items, self.max_weight, child2)


class BackpackGA:

    POPULATION_SIZE = 8
    ELITISM_PERCENT = 0.2

    def __init__(self, items, max_weight):
        self.items = items
        self.max_weight = max_weight
        self.population = None

    def solve(self, iterations=1000):
        self.population = [Chromosome(self.items, self.max_weight) for _ in range(self.POPULATION_SIZE)]
        for _ in range(iterations):
            self.perform_iteration()
        self.population.sort(key=lambda ch: ch.fitness, reverse=True)
        return self.population[0].get_phenotype()

    def perform_iteration(self):
        elite = self.select_elite()
        other = self.select_other()
        children = self.cross(other)
        self.mutate(children)

        self.population = [*elite, *children]

    def select_elite(self, ):
        self.population.sort(key=lambda ch: ch.fitness, reverse=True)
        elite_count = int(self.POPULATION_SIZE * self.ELITISM_PERCENT)
        return self.population[:elite_count]

    def select_other(self):
        selected = []

        fitnesses = [ch.fitness for ch in self.population]
        fitnesses_cumulated = np.cumsum(fitnesses).tolist()

        other_count = self.POPULATION_SIZE - int(self.POPULATION_SIZE * self.ELITISM_PERCENT)

        for _ in range(other_count):
            rand = random() * fitnesses_cumulated[-1]
            index = len(list(takewhile(lambda f: f < rand, fitnesses_cumulated)))
            selected.append(self.population[index])

        return selected

    def cross(self, chromosomes):
        if len(chromosomes) % 2 != 0:
            chromosomes.append(copy(chromosomes[0]))

        parents = [chromosomes[i:i+2] for i in range(0, len(chromosomes), 2)]
        childs = []

        for parent_1, parent_2 in parents:
            childs.extend(parent_1.cross(parent_2))

        return childs

    def mutate(self, chromosomes):
        for chromosome in chromosomes:
            chromosome.mutate()


if __name__ == '__main__':

    test_items = [
        Item(3, 266),
        Item(13, 442),
        Item(10, 671),
        Item(9, 526),
        Item(7, 388),
        Item(1, 245),
        Item(8, 210),
        Item(8, 145),
        Item(2, 126),
        Item(9, 322),
    ]

    test_max_weight = 35

    gs = BackpackGA(test_items, test_max_weight)
    items = gs.solve(iterations=1000)
    total_value = sum([item.value for item in items])
    total_weight = sum([item.weight for item in items])

    print(items)
    print('total value =', total_value)
    print('total weight =', total_weight)

    assert total_value == 2222
    assert total_weight == 32
