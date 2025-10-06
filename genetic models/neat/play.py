import cv2
import gym
import neat
import pickle
import warnings
import visualize
import gym.logger
import numpy as np
import multiprocessing
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
gym.logger.set_level(40)

TRAINING_LEVEL_LIST = ["1-2", "1-4", "2-1", "2-3","3-2", "3-4", "4-1", "4-3","5-1", "5-4", "6-2", "6-4","7-1", "8-2"]

class fitness(object):

    def __init__(self, genome, config):
        
        self.genome = genome
        self.config = config
        
    def eval(self):

        total_fitness = 0
        num_levels = 1

        for _ in range(num_levels):
            level_fitness = self.evaluate_single_level()
            total_fitness += level_fitness
            
        return total_fitness // num_levels

    def evaluate_single_level(self):

        lvl = np.random.choice(TRAINING_LEVEL_LIST)
        env_id = f'SuperMarioBros-{lvl}-v1'
        env = gym.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        try:
            obs = env.reset()
            done = False
            patience = 0
            max_x_pos = 0

            while not done:
                obs_processed = self.preprocess_observation(obs)
                
                output = net.activate(obs_processed)
                action = output.index(max(output))
                obs, reward, done, info = env.step(action)

                current_x_pos = info.get("x_pos", 0)
                if current_x_pos > max_x_pos:
                    max_x_pos = current_x_pos
                    patience = 0
                
                else:
                    patience += 1
                
                if patience > 100:
                    break
                    
                if info.get("flag_get", False):

                    fitness_score = max_x_pos + 1000
                    env.close()
                    return fitness_score

            fitness_score = max_x_pos
                
            env.close()
            return float(fitness_score)
            
        except Exception as e:

            print(f"Error en evaluación: {e}")
            env.close()
            return 0

    def preprocess_observation(self, obs):

        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(obs, (32, 32))
        screen = np.ndarray.flatten(screen)

        return screen
            
def eval_genomes(genome, config):
    
    eval = fitness(genome, config)
    return eval.eval()


if __name__ == '__main__':

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'neat-config-feedforward.txt')


    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(generation_interval=50, filename_prefix='checkpoints/neat-checkpoint-'))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count()-1, eval_genomes)

    winner = pop.run(pe.evaluate, 200)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg", smooth_best=True, window_size=10)

    with open("winner-feedforward","wb") as filename:
        pickle.dump(winner,filename)

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)