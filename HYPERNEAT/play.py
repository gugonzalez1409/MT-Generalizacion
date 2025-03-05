from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation
from pureples.shared.visualize import draw_net
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.gym_runner import run_hyper
from pureples.shared.substrate import Substrate
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import multiprocessing
import numpy as np
import gym.logger
import pickle
import logging
import neat
import gym



gym.logger.set_level(40)

"""
Run hyper
"""


def eval_parallel(genome, config, env_maker, substrate, max_steps, activations):
    env = env_maker()
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    net = create_phenotype_network(cppn, substrate)

    fitnesses = []
    for _ in range(10):
        ob = env.reset()
        net.reset()
        total_reward = 0

        for _ in range(max_steps):
            #env.render()
            for _ in range(activations):
                ob = np.ndarray.flatten(ob)
                ob = ob / 255.0 * 2 - 1
                o = net.activate(ob)
            noise = np.random.normal(0, 0.1, len(o))
            action = np.argmax(o + noise)
            ob, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        fitnesses.append(total_reward)
    
    return np.mean(fitnesses)

"""
Definicion del substrate

"""

input_width, input_height = 8, 8
coord_input = []
for y in range(input_height):
    for x in range(input_width):
        x_coord = (x / input_width) * 2 - 1
        y_coord = (y / input_height) * 2 - 1
        coord_input.append((x_coord, y_coord))

coord_ocultas = []
size_ocultas = [(6,6), (4,4), (4,4)]
z_coords = [-0.5, 0 , 0.5]
for layer_idx, (width, height) in enumerate(size_ocultas):
    layer = []
    z = z_coords[layer_idx]
    for y in range(height):
        for x in range(width):
            x_coord = (x/(width - 1 )) * 2 - 1
            y_coord = (y/(height -1 )) * 2 - 1
            layer.append((x_coord, y_coord))
    coord_ocultas.append(layer)

coord_output = [(i / (len(SIMPLE_MOVEMENT) - 1) * 2 - 1, 1.0) for i in range(len(SIMPLE_MOVEMENT))]

substrate = Substrate(coord_input, coord_output, coord_ocultas)

"""
Crea el entorno de gym en escala de grises y observaciones de 8x8

"""

def make_env():

    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=8)

    return env



gens = 50 # numero de generaciones
max_steps = 2000
activations = 3 # activaciones de red CPPN

"""
Creacion del config (NEAT-Python)

"""

CONFIG = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward'
)

"""
HyperNeat para entornos de gym (PUREPLES)

"""
def run(gens, env_maker):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    def eval_genomes(genomes, config):
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(pool.apply_async(eval_parallel, (genome, config, env_maker, substrate, max_steps, activations)))

        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get()

    pop = neat.Population(CONFIG)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, gens)
    return winner, stats



if __name__ == '__main__':

    multiprocessing.freeze_support()

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)

    WINNER = run(200, make_env)[0]

    cppn = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NET = create_phenotype_network(cppn, substrate)
    draw_net(cppn, filename="hyperneat_mario_cppn")
    draw_net(NET, filename="hyperneat_mario_winner")
    with open('hyperneat_mario_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

