import math
from copy import copy
from itertools import takewhile
from random import randint, random, shuffle
import numpy as np


class City:

    next_id = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.id = City.next_id
        City.next_id += 1

    def __repr__(self):
        return str(self.id)

    def distance_to(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)


class Chromosome:

    MUTATION_PROBABILITY = 0.01

    def __init__(self, cities, gens=None):
        self.cities = cities
        self.gens = gens if gens is not None else self.random_gens()
        self.fitness = self.calc_fitness()

    def random_gens(self):
        gens = list(range(len(self.cities)))
        shuffle(gens)
        return gens

    def get_phenotype(self):
        cities = []
        for gen in self.gens:
            cities.append(self.cities[gen])
        return cities

    def calc_fitness(self):
        return 1 / self.calc_distance()

    def calc_distance(self):
        cities = self.get_phenotype()
        prev_city = cities[0]
        distance = 0

        for city in cities[1:]:
            distance += prev_city.distance_to(city)
            prev_city = city

        distance += cities[0].distance_to(cities[-1])
        return distance

    def cross(self, other):
        child = copy(self.gens)
        index = len(self.gens) // 2
        for gen in other.gens:
            if gen not in child[0:index]:
                child[index] = gen
                index += 1
        return Chromosome(self.cities, child)

    def mutate(self):
        for pos1 in range(len(self.gens)):
            if random() < self.MUTATION_PROBABILITY:

                pos2 = pos1
                while pos2 == pos1:
                    pos2 = randint(0, len(self.gens) - 1)

                temp = self.gens[pos1]
                self.gens[pos1] = self.gens[pos2]
                self.gens[pos2] = temp

                self.fitness = self.calc_fitness()


class SalesmanGA:

    POPULATION_SIZE = 100
    ELITISM_PERCENT = 0.2

    def __init__(self):
        self.population = None

    def solve(self, cities, iterations=1000):
        self.population = [Chromosome(cities) for _ in range(self.POPULATION_SIZE)]
        for i in range(iterations):
            self.perform_iteration()
            if i % 1000 == 0:
                self.population.sort(key=lambda ch: ch.fitness, reverse=True)
                print(self.population[0].get_phenotype(), self.population[0].calc_distance())
        self.population.sort(key=lambda ch: ch.fitness, reverse=True)
        return self.population[0].get_phenotype(), self.population[0].calc_distance()

    def perform_iteration(self):
        self.population.sort(key=lambda ch: ch.fitness, reverse=True)

        elite_count = int(self.POPULATION_SIZE * self.ELITISM_PERCENT)
        next_population = self.population[:elite_count]

        fitnesses = [ch.fitness for ch in self.population]
        fitnesses_cumulated = np.cumsum(fitnesses).tolist()

        while len(next_population) < self.POPULATION_SIZE:

            # Select parents with roulette
            rand1 = random() * fitnesses_cumulated[-1]
            rand2 = random() * fitnesses_cumulated[-1]

            parent1 = self.population[len(list(takewhile(lambda f: f < rand1, fitnesses_cumulated)))]
            parent2 = self.population[len(list(takewhile(lambda f: f < rand2, fitnesses_cumulated)))]

            # Cross parents
            child = parent1.cross(parent2)

            # Mutate child
            child.mutate()

            next_population.append(child)

        # Replace population
        self.population = next_population


if __name__ == '__main__':

    test_cities = [
        City(119, 38),
        City(37, 38),
        City(197, 55),
        City(85, 165),
        City(12, 50),
        City(100, 53),
        City(81, 142),
        City(121, 137),
        City(85, 145),
        City(80, 197),
        City(91, 176),
        City(106, 55),
        City(123, 57),
        City(40, 81),
        City(78, 125),
        City(190, 46),
        City(187, 40),
        City(37, 107),
        City(17, 11),
        City(67, 56),
        City(78, 133),
        City(87, 23),
        City(184, 197),
        City(111, 12),
        City(66, 178),
    ]

    ga = SalesmanGA()
    ordered_cities, distance = ga.solve(test_cities, iterations=100000)
    print(ordered_cities)
    print('Distance =', distance)
