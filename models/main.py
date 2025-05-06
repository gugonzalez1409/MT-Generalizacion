import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
from models.utils.trainers import trainDQN, trainPPO, trainRecurrentPPO, trainRainbow


"""
Agentes para generalizacion en SMB

SB3 v1.6
Gym v0.21.0
Gym SMB v7.4.0
Pytorch 2.6.0

"""


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorized', action='store_true', default=False, help='Activa entrenamiento con entorno vectorizado')
    parser.add_argument('--algo', type=str, choices=['PPO', 'DQN', 'RPPO', 'RDQN'], help= 'Algoritmo a entrenar')
    parser.add_argument('--explore', type=int, help= 'Pasos de exploracion en ExploreGo')
    parser.add_argument('--random', action='store_true', default=False, help= 'Cantidad de frames en las que se usa randomizacion de entorno')
    parser.add_argument('--custom', action='store_true', default=False, help='Activa recompensa personalizada')

    args = parser.parse_args()

    if args.algo == None:
        parser.error('Debes especificar un algoritmo para entrenar')
        
    if args.algo == 'PPO':
        trainPPO(args.explore, args.random, args.custom, args.vectorized)

    if args.algo == 'DQN':
        trainDQN(args.explore, args.random, args.custom, args.vectorized)

    if args.algo == 'RPPO':
        trainRecurrentPPO(args.explore, args.random, args.custom, args.vectorized)

    if args.algo == 'RDQN':
        trainRainbow(args.explore, args.random, args.custom, args.vectorized)