import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
import neat
import pickle
import csv
import cv2
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


"""
Entorno para evaluacion de modelo en SMB
el agente tiene 10 intentos a cada nivel del juego
donde se mide la recompensa obtenida y cuanto del nivel se logró completar

"""

# 
EVALUATION_LEVEL_LIST = ['1-1', '1-3', '2-4', '3-1','3-3', '4-2', '5-2', '5-3', '6-1', '6-3', '8-1', '7-3', '8-3']

TRAINING_LEVEL_LIST = ['1-2', '1-4', '2-1', '2-3', '3-2', '3-4', '4-1', '4-3', '5-1', '5-4', '6-2', '6-4', '7-1', '8-2']

# lista con la maxima x_pos de cada nivel, para evitar usar estimado

stageLengthMap = {

    (1, 1): 3376,
    (1, 2): 3072,
    (1, 3): 2624,
    (1, 4): 2560,
    (2, 1): 3408,
    (2, 3): 3792,
    (2, 4): 2560,
    (3, 1): 3408,
    (3, 2): 3552,
    (3, 3): 2608,
    (3, 4): 2560,
    (4, 1): 3808,
    (4, 2): 3584,
    (4, 3): 2544,
    (5, 1): 3392,
    (5, 2): 3408,
    (5, 3): 2624,
    (5, 4): 2560,
    (6, 1): 3216,
    (6, 2): 3664,
    (6, 3): 2864,
    (6, 4): 2560,
    (7, 1): 3072,
    (7, 3): 3792,
    (8, 1): 6224,
    (8, 2): 3664,
    (8, 3): 3664,
}


def process_obs(obs):

    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(obs, (32, 32))
    screen = np.ndarray.flatten(screen)

    return screen


def predict(net, obs):

    obs = process_obs(obs)
    output = net.activate(obs)
    action = np.argmax(output)

    return action


config_path = 'neat-config-feedforward'
winner_path = 'winner-feedforward'


# hacer red de neat
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


with open(winner_path, "rb") as f:
    winner = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(winner, config)

levels = [f"SuperMarioBros-{lvl}-v1" for lvl in EVALUATION_LEVEL_LIST]
keys = EVALUATION_LEVEL_LIST.copy()

csv_filename = 'NEAT_evaluation.csv'

with open(csv_filename, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['level', 'avg_reward', 'std_reward','avg_completion', 'max_completion'])

    results = {}
    for i, level in enumerate(levels):

        env = gym.make(level)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = customReward(env)

        total_rewards = []
        completition_rates = []
        max_completion = 0

        level_key = keys[i]
        world, stage = map(int, level_key.split('-'))
        lvl_length = stageLengthMap.get((world, stage), 3000)

        try:
            for j in range(10):

                obs = env.reset()
                total_reward = 0
                max_x_pos = 0
                num_envs = 1

                while True:

                    action = predict(net, obs)
                    obs, reward, done, info = env.step(action)
                    
                    total_reward += reward
                    max_x_pos = max(max_x_pos, info['x_pos'])

                    env.render()

                    if done:
                        if info['flag_get'] == True:
                            lvl_length = info['x_pos']
                            print("nivel completado: ", keys[i])

                        break


                total_rewards.append(total_reward)
                completition_rate = (max_x_pos / lvl_length) * 100
                completition_rates.append(completition_rate)
                max_completion = max(max_completion, completition_rate)

        finally:
            env.close()


        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_completition = np.mean(completition_rates)

        results[keys[i]] = { # guardar resultados a csv
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_completition": avg_completition,
            "max_completion": max_completion
        }

        writer.writerow([level, avg_reward, std_reward,avg_completition, max_completion])


# graficar resultados
sns.set_theme(style="whitegrid")

data = []
for lvl in keys:
    data.append({
        "Nivel": lvl,
        "Recompensa Promedio": results[lvl]['avg_reward'],
        "Desviación Estándar": results[lvl]['std_reward'],
        "Completado Promedio": results[lvl]['avg_completition'],
        "Completado Máximo": results[lvl]['max_completion']
    })
df = pd.DataFrame(data)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.barplot(
    data=df,
    x="Nivel",
    y="Recompensa Promedio",
    ax=axes[0],
    color="blue",
    saturation=0.5,
    errorbar=None,
    alpha=.6,
)

x_positions = np.arange(len(df))
for i, (mean, std) in enumerate(zip(df["Recompensa Promedio"], df["Desviación Estándar"])):
    axes[0].errorbar(i, mean, yerr=std, fmt='none', c='black', capsize=5)

axes[0].set_title("Recompensa por Nivel")
axes[0].set_ylabel("Recompensa")
axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(df["Nivel"])
axes[0].set_ylim(-500, 4000)

niveles = df["Nivel"]
promedios = df["Completado Promedio"]
maximos = df["Completado Máximo"]

axes[1].bar(niveles, promedios, label="Promedio completado", color="steelblue", alpha=0.7)
axes[1].bar(niveles, maximos - promedios, bottom=promedios, label="Máximo completado", color="orange", alpha=0.7)

axes[1].set_title("Completado por nivel")
axes[1].set_ylabel("Porcentaje de nivel superado")
axes[1].set_ylim(0, 100)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(csv_filename), 'eval_results.png'))
plt.show()