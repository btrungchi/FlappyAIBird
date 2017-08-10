"""
"""

import os
import sys
import pickle
import numpy as np
from FlappyBirdClone.flappy import init_game
from models import EvolNeuralNet
from models import next_generation

GENERATION_POPULATION = 150


class EvolNetTrainer(object):
    def __init__(self):
        self.generation = [EvolNeuralNet() for _ in xrange(GENERATION_POPULATION)]
        self.cur_index = 0
        self.n_generation = 1
    
    def is_cur_individual_flap(self, nearest_pipe_y, nearest_pipe_y2, nearest_pipe_dist):
        cur_individual = self.generation[self.cur_index]
        decision = cur_individual.forward(np.array([nearest_pipe_y, nearest_pipe_y2, nearest_pipe_dist]))
        if decision > 0.5:
            return True
        else:
            return False
    
    def cur_individual_dead(self, fitness, score):
        sys.stdout.write(str(score) + " ")
        cur_individual = self.generation[self.cur_index]
        cur_individual.fitness = fitness
        if self.cur_index < GENERATION_POPULATION - 1:
            self.cur_index += 1
        else:
            print "\n:::::: GENERATION %d ENDED" %(self.n_generation)
            pickle.dump(self, open('../checkpoint.pickle', 'wb'))
            self.generation = next_generation(self.generation, GENERATION_POPULATION)
            self.cur_index = 0
            self.n_generation += 1

def main(args):
    if len(args) > 1 and os.path.isfile(args[1]):
        trainer = pickle.load(open(args[1], 'rb'))
    else:
        trainer = EvolNetTrainer()

    os.chdir(os.getcwd() + '/FlappyBirdClone')
    init_game(trainer)


if __name__ == "__main__":
    main(sys.argv)