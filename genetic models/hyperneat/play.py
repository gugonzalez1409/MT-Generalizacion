
import gym
import neat
import pickle
import logging
import gym_super_mario_bros
from gym_utils import SMBRamWrapper
from nes_py.wrappers import JoypadSpace
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_hyper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from pureples.hyperneat.hyperneat import create_phenotype_network
from hyperneat_mario import run_hyper_mario
import matplotlib.pyplot as plt
#from gym.wrappers import GrayScaleObservation, NormalizeObservation, FlattenObservation


"""
Definicion del substrate

"""

input_width, input_height = 13, 16
coord_input = []
for y in range(input_height):
    for x in range(input_width):
        x_coord = (x / (input_width - 1)) * 2 - 1
        y_coord = (y / (input_height - 1)) * 2 - 1

        coord_input.append((x_coord, y_coord))

#coord_ocultas = []
#size_ocultas = [(4,4)]

"""for layer_idx, (width, height) in enumerate(size_ocultas):
    layer = []
    for y in range(height):
        for x in range(width):
            x_coord = (x/(width - 1 )) * 2 - 1
            y_coord = (y/(height -1 )) * 2 - 1
            layer.append((x_coord, y_coord))
    coord_ocultas.append(layer)"""

#coord_output = [(i / (len(SIMPLE_MOVEMENT) - 1) * 2 - 1, 1.0) for i in range(len(SIMPLE_MOVEMENT))]

coord_output = [(x, y) for x, y in coord_input]

substrate = Substrate(coord_input, coord_output)

"""
Crea el entorno de gym en escala de grises

"""

def make_env():

    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    x0 = 0
    x1 = 16
    y0 = 0
    y1 = 13
    n_stack = 1
    n_skip = 1
    env = SMBRamWrapper(env, [x0, x1, y0, y1], n_stack=n_stack, n_skip=n_skip)

    print("obs shape: ", env.observation_space.shape)

    return env



gens = 150 # numero de generaciones
max_steps = 3000
activations = 1 # activaciones de red CPPN

"""
Creacion del config (NEAT-Python)

"""

CONFIG = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'hyper-neat-config-feedforward'
)


def run(gens, env):

    winner, stats = run_hyper_mario(
        gens=gens, env=env, max_steps=max_steps, config= CONFIG, substrate=substrate, activations=activations, input_shape=(13,16))

    return winner, stats

if __name__ == '__main__':

    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)


    WINNER, STATS = run(150, make_env())

    #cppn = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    #NET = create_phenotype_network(cppn, substrate)
    #draw_net(cppn, filename="hyperneat_mario_cppn")
    #draw_net(NET, filename="hyperneat_mario_winner")
    #with open('hyperneat_mario_cppn.pkl', 'wb') as output:
    #    pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

