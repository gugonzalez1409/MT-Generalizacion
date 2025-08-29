import gym.logger
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from reward import customReward
import gym_super_mario_bros
import gym
import numpy as np
import neat
import pickle
import cv2
import visualize
import multiprocessing
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
gym.logger.set_level(40)

TRAINING_LEVEL_LIST = ["1-2", "1-4", "2-1", "2-3","3-2", "3-4", "4-1", "4-3","5-1", "5-4", "6-2", "6-4","7-1", "8-2"]

class fitness(object):

    def __init__(self, genome, config):

        self.genome = genome
        self.config = config
        
    def eval(self):

        lvl = np.random.choice(TRAINING_LEVEL_LIST)
        env_id = f'SuperMarioBros-{lvl}-v1'
        self.env = gym.make(env_id)
        self.env = customReward(self.env)
        #self.env = ExploreGo(self.env, exploration_steps=200, explorer=None)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        net = neat.nn.RecurrentNetwork.create(self.genome, self.config)

        obs = self.env.reset()
        done = False
        fitness_total = 0
        patience = 0
        max_x_pos = 0

        while done==False:
            
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(obs, (32, 32))
            screen = np.ndarray.flatten(screen)
            #self.env.render()
            output = net.activate(screen)
            action = output.index(max(output))
            
            obs, reward, done, info = self.env.step(action)

            fitness_total += reward

            if info["x_pos"] > max_x_pos:
                
                max_x_pos = info["x_pos"]
                patience = 0

            else:
                patience += 1
            
            if done or patience > 100:
                break

            if info["flag_get"]:
                fitness_total += 10000
                break

        return int(fitness_total)
            
            
        
def eval_genomes(genome, config):
    
    eval = fitness(genome, config)
    return eval.eval()


if __name__ == '__main__':

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'neat-config-feedforward')


    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(generation_interval=50))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count()-1, eval_genomes)

    winner = pop.run(pe.evaluate, 200)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    with open("winner-feedforward","wb") as filename:
        pickle.dump(winner,filename)

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)