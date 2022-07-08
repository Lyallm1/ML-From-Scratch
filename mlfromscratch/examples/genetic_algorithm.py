from mlfromscratch.unsupervised_learning import GeneticAlgorithm

genetic_algorithm = GeneticAlgorithm('Genetic Algorithm', 100, 0.05)
print("")
print("+--------+")
print("|   GA   |")
print("+--------+")
print("Description: Implementation of a Genetic Algorithm which aims to produce")
print("the user specified target string. This implementation calculates each")
print("candidate's fitness based on the alphabetical distance between the candidate")
print("and the target. A candidate is selected as a parent with probabilities proportional")
print("to the candidate's fitness. Reproduction is implemented as a single-point")
print("crossover between pairs of parents. Mutation is done by randomly assigning")
print("new characters with uniform probability.")
print("")
print("Parameters")
print("----------")
print("Target String: 'Genetic Algorithm'")
print("Population Size: 100")
print("Mutation Rate: 0.05")
print("")
genetic_algorithm.run(1000)
