import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gym
import csv
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from nes_py.wrappers import JoypadSpace
from icm.reward import customReward
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.atari_wrappers import AtariWrapper

"""
Entorno para evaluacion de modelo en SMB
el agente tiene 10 intentos a cada nivel del juego
donde se mide la recompensa obtenida y cuanto del nivel se logró completar

"""


path = r'./statistics/log_dir/basePPO/best_model.zip'
model = PPO.load(path)

levels = [f"SuperMarioBros-{w}-{s}-v0" for w in range(1,9) for s in range(1,5)]
keys = [f"{w}-{s}" for w in range(1,9) for s in range(1,5)]


csv_filename = 'PPOevaluation.csv'

with open(csv_filename, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['level', 'avg_reward', 'avg_completion', 'max_completion'])

    results = {}
    for i, level in enumerate(levels):

        print(f"Evaluando en {level}")

        env = gym.make(level)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = customReward(env)
        env = AtariWrapper(env= env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, clip_reward=False)

        total_rewards = []
        completition_rates = []
        max_completion = 0
        lvl_length = 3000 # valor aproximado, en caso de nunca llegar al final


        for j in range(10):

            obs = env.reset()
            total_reward = 0
            max_x_pos = 0

            while True:

                env.render()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                max_x_pos = max(max_x_pos, info['x_pos'])

                if done:
                    if info['flag_get'] == True:
                        lvl_length = info['x_pos']
                        print("x_pos final: ", lvl_length)
                    break


            total_rewards.append(total_reward)
            completition_rate = (max_x_pos / lvl_length) * 100
            completition_rates.append(completition_rate)
            max_completion = max(max_completion, completition_rate)

        env.close()


        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_completition = sum(completition_rates) / len(completition_rates)

        results[keys[i]] = {
            "avg_reward": avg_reward,
            "avg_completition": avg_completition,
            "max_completion": max_completion
        }

        writer.writerow([level, avg_reward, avg_completition, max_completion])


levels = list(results.keys())
avg_rewards = [results[l]['avg_reward'] for l in levels]
avg_completions = [results[l]['avg_completition'] for l in levels]
max_completions = [results[l]['max_completion'] for l in levels]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(levels, avg_rewards, color='blue')
plt.xticks(rotation=90)
plt.ylabel('Avg reward')
plt.title('Recompensa promedio por nivel')

plt.subplot(1, 2, 2)
plt.bar(levels, avg_completions, color='green', label='Promedio')
plt.bar(levels, max_completions, color='red', alpha=0.5, label='Maximo')
plt.xticks(rotation=90)
plt.ylabel('Completion %')
plt.title('Completado por Nivel')
plt.legend()

plt.tight_layout()
plt.savefig('eval_results.png')
plt.show()
