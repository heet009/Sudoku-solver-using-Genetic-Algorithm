import streamlit as st
import random as rndm
import time

# Define your genetic algorithm functions here
def make_gene(initial=None):
    if initial is None:
        initial = [0] * 9
    mapp = {}
    gene = list(range(1, 10))
    rndm.shuffle(gene)
    for i in range(9):
        mapp[gene[i]] = i
    for i in range(9):
        if initial[i] != 0 and gene[i] != initial[i]:
            temp = gene[i], gene[mapp[initial[i]]]
            gene[mapp[initial[i]]], gene[i] = temp
            mapp[initial[i]], mapp[temp[0]] = i, mapp[initial[i]]
    return gene

def make_chromosome(initial=None):
    if initial is None:
        initial = [[0] * 9] * 9
    chromosome = []
    for i in range(9):
        chromosome.append(make_gene(initial[i]))
    return chromosome

def make_population(count, initial=None):
    if initial is None:
        initial = [[0] * 9] * 9
    population = []
    for _ in range(count):
        population.append(make_chromosome(initial))
    return population

def get_fitness(chromosome):
    fitness = 0
    for i in range(9): # For each column
        seen = {}
        for j in range(9): # Check each cell in the column
            if chromosome[j][i] in seen:
                seen[chromosome[j][i]] += 1
            else:
                seen[chromosome[j][i]] = 1
        for key in seen: # Subtract fitness for repeated numbers
            fitness -= (seen[key] - 1)
    for m in range(3): # For each 3x3 square
        for n in range(3):
            seen = {}
            for i in range(3 * n, 3 * (n + 1)):  # Check cells in 3x3 square
                for j in range(3 * m, 3 * (m + 1)):
                    if chromosome[j][i] in seen:
                        seen[chromosome[j][i]] += 1
                    else:
                        seen[chromosome[j][i]] = 1
            for key in seen: # Subtract fitness for repeated numbers
                fitness -= (seen[key] - 1)
    return fitness

def crossover(ch1, ch2):
    new_child_1 = []
    new_child_2 = []
    for i in range(9):
        x = rndm.randint(0, 1)
        if x == 1:
            new_child_1.append(ch1[i])
            new_child_2.append(ch2[i])
        elif x == 0:
            new_child_2.append(ch1[i])
            new_child_1.append(ch2[i])
    return new_child_1, new_child_2

def mutation(ch, pm, initial):
    for i in range(9):
        x = rndm.randint(0, 100)
        if x < pm * 100:
            ch[i] = make_gene(initial[i])
    return ch

def read_puzzle(address):
    puzzle = []
    f = open(address, 'r')
    for row in f:
        temp = row.split()
        puzzle.append([int(c) for c in temp])
    return puzzle

def r_get_mating_pool(population): # Roulette wheel selection
    fitness_list = []
    pool = []
    for chromosome in population:
        fitness = get_fitness(chromosome)
        fitness_list.append((fitness, chromosome))
    fitness_list.sort()
    weight = list(range(1, len(fitness_list) + 1))
    for _ in range(len(population)):
        ch = rndm.choices(fitness_list, weight)[0]
        pool.append(ch[1])
    return pool

def get_offsprings(population, initial, pm, pc):
    new_pool = []
    i = 0
    while i < len(population):
        ch1 = population[i]
        ch2 = population[(i + 1) % len(population)]
        x = rndm.randint(0, 100)
        if x < pc * 100:
            ch1, ch2 = crossover(ch1, ch2)
        new_pool.append(mutation(ch1, pm, initial))
        new_pool.append(mutation(ch2, pm, initial))
        i += 2
    return new_pool


# Population size
DEFAULT_POPULATION = 1000

# Number of generations
DEFAULT_REPETITION = 1000

# Default probability of mutation
DEFAULT_PM = 0.1

# Default probability of crossover
DEFAULT_PC = 0.95

def main():
    st.title("Genetic Algorithm: Sudoku Solver")

    # Sidebar for sliders
    st.sidebar.header("Genetic Algorithm Parameters")
    population_size = st.sidebar.slider("Population Size", min_value=10, max_value=5000, value=DEFAULT_POPULATION)
    repetition = st.sidebar.slider("Number of Generations", min_value=10, max_value=5000, value=DEFAULT_REPETITION)
    pm = st.sidebar.slider("Probability of Mutation", min_value=0.0, max_value=1.0, value=DEFAULT_PM, step=0.01)
    pc = st.sidebar.slider("Probability of Crossover", min_value=0.0, max_value=1.0, value=DEFAULT_PC, step=0.01)
    st.write("Sudoku Puzzle:")
    st.dataframe(read_puzzle("Test2.txt"))
    # Run the genetic algorithm
    if st.button("Run Genetic Algorithm"):
        st.write("Running Genetic Algorithm...")
        tic = time.time()
        solved_sudoku = genetic_algorithm(population_size, repetition, pm, pc)
        toc = time.time()
        st.write("Genetic Algorithm Finished!")
        st.write("Time taken: ", toc - tic)
        st.write("Fitness of the solution: ", get_fitness(solved_sudoku))
        st.write("Solved Sudoku:")
        st.dataframe(solved_sudoku)
        

def genetic_algorithm(population_size, repetition, pm, pc):
    initial_file = "Test2.txt"
    initial = read_puzzle(initial_file)
    population = make_population(population_size, initial)
    for _ in range(repetition):
        mating_pool = r_get_mating_pool(population)
        rndm.shuffle(mating_pool)
        population = get_offsprings(mating_pool, initial, pm, pc)
        fit = [get_fitness(c) for c in population]
        m = max(fit)
        if m == 0:
            return population[0]
    return population[0]

if __name__ == "__main__":
    main()
