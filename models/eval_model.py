import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
import csv
import numpy as np
import matplotlib.pyplot as plt
from .utils.reward import customReward
from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO
from .rainbow.rainbow import Rainbow
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor, VecTransposeImage


EVALUATION_LEVEL_LIST = ['1-1', '1-3', '2-4', '3-1', '3-3', '4-2', '5-2', '5-3', '6-1', '6-3', '8-1', '8-3', '7-3']

TRAINING_LEVEL_LIST = ['1-2', '1-4', '2-1', '2-3', '3-2', '3-4', '4-1', '4-3', '5-1', '5-4', '6-2', '6-4', '7-1', '8-2']

"""
Entorno para evaluacion de modelo en SMB
el agente tiene 10 intentos a cada nivel del juego
donde se mide la recompensa obtenida y cuanto del nivel se logró completar

"""

path = 'models/RDQN_random_mario'
model = Rainbow.load(path)

levels = [f"SuperMarioBros-{lvl}-v1" for lvl in EVALUATION_LEVEL_LIST]
keys = EVALUATION_LEVEL_LIST.copy()

csv_filename = 'RDQN_random_evaluation.csv'

with open(csv_filename, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['level', 'avg_reward', 'std_reward','avg_completion', 'max_completion'])

    results = {}
    for i, level in enumerate(levels):


        env = gym.make(level)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        env = MaxAndSkipEnv(env, skip=4)
        env = WarpFrame(env)

        env = customReward(env)

        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4, channels_order='last')
        env = VecTransposeImage(env)

        #env = VecVideoRecorder(env, f'statistics/videos/{level}', record_video_trigger=lambda x: True, name_prefix=f'PPO{level}', video_length=20000)
        env = VecMonitor(env)

        total_rewards = []
        completition_rates = []
        max_completion = 0
        lvl_length = 3000 # valor aproximado, en caso de nunca llegar al final

        try:
            for j in range(10):

                obs = env.reset()
                total_reward = 0
                max_x_pos = 0
                #lstm_states = None
                num_envs = 1

                #episode_starts = np.ones((num_envs,), dtype=bool)
                while True:

                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, info = env.step(action)
                    episode_starts = done
                    total_reward += reward[0]
                    max_x_pos = max(max_x_pos, info[0]['x_pos'])

                    env.envs[0].render()

                    if done[0]:
                        if info[0]['flag_get'] == True:
                            lvl_length = info[0]['x_pos']
                            print("x_pos final: ", lvl_length)

                        break


                total_rewards.append(total_reward)
                completition_rate = (max_x_pos / lvl_length) * 100
                completition_rates.append(completition_rate)
                max_completion = max(max_completion, completition_rate)

        finally:
            env.close()


        avg_reward = sum(total_rewards) / len(total_rewards)
        std_reward = np.std(total_rewards)
        avg_completition = sum(completition_rates) / len(completition_rates)

        results[keys[i]] = { # guardar results para csv
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_completition": avg_completition,
            "max_completion": max_completion
        }

        writer.writerow([level, avg_reward, std_reward,avg_completition, max_completion])

    # graficar resultados

    # recompensa y desviacion estandar
    levels_plot = list(results.keys())
    avg_rewards = [results[l]['avg_reward'] for l in levels_plot]
    avg_completions = [results[l]['avg_completition'] for l in levels_plot]
    max_completions = [results[l]['max_completion'] for l in levels_plot]
    std_rewards = [results[l]['std_reward'] for l in levels_plot]

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    x = np.arange(len(levels_plot)) * 1.5
    width = 0.4
    bar_avg = plt.bar(x - (width/2 + 0.02), avg_rewards, width, color='blue', label='Promedio')
    bar_std = plt.bar(x + (width/2 + 0.02), std_rewards, width, color='orange', alpha=0.7, label='Desviación Estándar')


    plt.xticks(x, levels_plot, rotation=90)
    plt.ylabel('Recompensa')
    plt.title('Recompensa por Nivel')
    plt.legend()

    # avance por nivel y máximo avanzado

    plt.subplot(1, 2, 2)
    x = np.arange(len(levels_plot)) * 1.5
    width = 0.4
    bar_avg_comp = plt.bar(x - (width/2 + 0.02), avg_completions, width, color='green', label='Promedio')
    bar_max_comp = plt.bar(x + (width/2 + 0.02), max_completions, width, color='red', alpha=0.7, label='Maximo')

    plt.xticks(x, levels_plot, rotation=90)
    plt.ylabel('Porcentaje de nivel superado (estimado)')
    plt.title('Completado por Nivel')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(csv_filename), 'eval_results.png'))
    plt.show()