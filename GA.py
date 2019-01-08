import pandas as pd
import random
import tensorflow as tf
class Genetic_algorithm(object):
    def __init__(self):
        self.generations = 8 # Number of times to evolve the population.
        self.population = 5  # Number of genomes in each generation.
        self.fraction_best_kept = 0.5
        self.mutate = 0.3
        self.all_possible_genes={
            'learning_rate_g':[1e-4,2e-4],
            'learning_rate_d':[1e-4,2e-4],
            'update_rate':[1,2]
        }

    def initial_population(self):
        population = pd.DataFrame(columns = ["learning_rate_g", "learning_rate_d", "update_rate", "accuracy_score"])
        for i in range(self.population):
            for j in range(population.shape[1]-1):
                population.loc[i, population.columns[j]] = self.all_possible_genes[population.columns[j]][
                    random.randrange(len(self.all_possible_genes[population.columns[j]]))]
        return population

if __name__ == '__main__':
    GA = Genetic_algorithm()
    print(GA.initial_population())
