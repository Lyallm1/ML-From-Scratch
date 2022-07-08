from __future__ import print_function, division
import string, numpy as np

class GeneticAlgorithm():
    def __init__(self, target_string, population_size, mutation_rate):
        self.target = target_string
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.letters = [" "] + list(string.ascii_letters)

    def _initialize(self):
        self.population = []
        for _ in range(self.population_size): self.population.append(''.join(np.random.choice(self.letters, size=len(self.target))))

    def _calculate_fitness(self):
        population_fitness = []
        for individual in self.population:
            loss = 0
            for i in range(len(individual)): loss += abs(self.letters.index(individual[i]) - self.letters.index(self.target[i]))
            population_fitness.append(1 / (loss + 1e-6))
        return population_fitness

    def _mutate(self, individual):
        individual = list(individual)
        for j in range(len(individual)):
            if np.random.random() < self.mutation_rate: individual[j] = np.random.choice(self.letters)
        return "".join(individual)

    def _crossover(self, parent1, parent2):
        cross_i = np.random.randint(0, len(parent1))
        return parent1[:cross_i] + parent2[cross_i:], parent2[:cross_i] + parent1[cross_i:]

    def run(self, iterations):
        self._initialize()
        for epoch in range(iterations):
            population_fitness = self._calculate_fitness()
            fittest_individual = self.population[np.argmax(population_fitness)]
            if fittest_individual == self.target: break
            new_population = []
            for _ in np.arange(0, self.population_size, 2):
                parent1, parent2 = np.random.choice(self.population, size=2, p=[fitness / sum(population_fitness) for fitness in population_fitness], replace=False)
                new_population += [self._mutate(self._crossover(parent1, parent2)[0]), self._mutate(self._crossover(parent1, parent2)[1])]
            print("[%d Closest Candidate: '%s', Fitness: %.2f]" % (epoch, fittest_individual, max(population_fitness)))
            self.population = new_population
        print("[%d Answer: '%s']" % (epoch, fittest_individual))
