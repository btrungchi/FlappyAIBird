import numpy as np

_LAYER_SIZE_INPUT = 4
_LAYER_SIZE_HIDDEN = 2
_LAYER_SIZE_OUTPUT = 1

_CROSSOVER_RATE = .9

class ChromOp(object):
    """
    Operators on chromosome
    """

    @staticmethod
    def chrom_encode(w1, w2):
        return np.append(w1.flatten(), w2.flatten())

    @staticmethod
    def chrom_decode(chromosome):
        chromosome = np.array(chromosome)
        head_len = _LAYER_SIZE_INPUT * _LAYER_SIZE_HIDDEN
        chrom_head = chromosome[:head_len]
        chrom_tail = chromosome[head_len:]
        weight_1 = chrom_head.reshape((_LAYER_SIZE_INPUT, _LAYER_SIZE_HIDDEN))
        weight_2 = chrom_tail.reshape((_LAYER_SIZE_HIDDEN, _LAYER_SIZE_OUTPUT))
        return weight_1, weight_2

    @staticmethod
    def selection(arr_chrom_code, arr_chrom_fitness):
        roulette_max = np.sum(arr_chrom_fitness)
        spin_result = np.random.uniform(0, roulette_max)
        sumup = 0
        for i in range(0, len(arr_chrom_code)):
            if sumup + arr_chrom_fitness[i] > spin_result:
                return arr_chrom_code[i]
            sumup += arr_chrom_fitness[i]

    @staticmethod
    def crossover(chrom_code_1, chrom_code_2):
        min_chrom_size = min(chrom_code_1.size, chrom_code_2.size)
        cross_pos = np.random.randint(0, min_chrom_size - 1)
        new_chrom_code_1 = np.append(
            chrom_code_1[:cross_pos], chrom_code_2[cross_pos:])
        new_chrom_code_2 = np.append(
            chrom_code_2[:cross_pos], chrom_code_1[cross_pos:])
        return new_chrom_code_1, new_chrom_code_2

    @staticmethod
    def mutate(chrom_code, mutation_rate=.1):
        new_chromo = []
        for code in chrom_code:
            if np.random.random() < mutation_rate:
                code += np.random.normal(0, 0.5)
            new_chromo.append(code)
        return new_chromo

class EvolNeuralNet(object):
    def __init__(self, weights=None, chromosome=None):
        self.fitness = 1
        self.input_size = _LAYER_SIZE_INPUT
        self.hidden_size = _LAYER_SIZE_HIDDEN
        self.ouput_size = _LAYER_SIZE_OUTPUT

        if chromosome is not None:
            self.w1, self.w2 = ChromOp.chrom_decode(chromosome)
        else:
            if weights is not None:
                self.w1 = weights[0]
                self.w2 = weights[1]
            else:
                self.w1 = np.random.randn(self.input_size, self.hidden_size)
                self.w2 = np.random.randn(self.hidden_size, self.ouput_size)

        self.chromosome = ChromOp.chrom_encode(self.w1, self.w2)

    def forward(self, input_val):
        input_val = np.append(input_val, np.ones(1))
        z1 = np.dot(input_val, self.w1)
        output1 = EvolNeuralNet.activation(z1)
        z2 = np.dot(output1, self.w2)
        output2 = EvolNeuralNet.activation(z2)
        return output2

    @staticmethod
    def activation(z_value):
        return 1 / (1 + np.exp(-z_value))


def next_generation(neural_nets, n_population):
    new_generation = []
    arr_chrom_code = [net.chromosome for net in neural_nets]
    arr_chrom_fitness = [net.fitness for net in neural_nets]
    for _ in xrange(n_population/2):    # remove after selected?
        parent_chrom_code_1 = ChromOp.selection(arr_chrom_code, arr_chrom_fitness)
        parent_chrom_code_2 = ChromOp.selection(arr_chrom_code, arr_chrom_fitness)
        if np.random.uniform(0, 1.0) < _CROSSOVER_RATE:
            child_code_1, child_code_2 = ChromOp.crossover(parent_chrom_code_1, parent_chrom_code_2)
        else:
            child_code_1, child_code_2 = parent_chrom_code_1, parent_chrom_code_2
        child_code_1 = ChromOp.mutate(child_code_1)
        child_code_2 = ChromOp.mutate(child_code_2)
        new_generation.append(EvolNeuralNet(chromosome=child_code_1))
        new_generation.append(EvolNeuralNet(chromosome=child_code_2))
    return new_generation
