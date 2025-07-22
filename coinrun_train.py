import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from typing import Callable
import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecMonitor, VecExtractDictObs
from models.rainbow.rainbow import Rainbow
from models.rainbow.policies import RainbowPolicy
from models.generalization.ImpalaCNN import ImpalaCNN
from models.generalization.ExploreGo import ExploreGo
from procgen import ProcgenEnv


tensorboard_log = r'./procgen_models/tensorboard_log/'
log_dir = r'./procgen_models/log_dir/'


def linear_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:

        return progress_remaining * initial_value

    return func



def make_env(train= True, algo='ppo'):

    num_levels = 500
    start_level = 0 if train else 1000

    if algo == 'ppo':
        num_envs = 64
    
    else:
        num_envs = 128

    env = ProcgenEnv(
        num_envs=num_envs,
        env_name="coinrun",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode="hard"
    )
    env = VecExtractDictObs(env, "rgb")
    env = VecMonitor(env)

    return env


def trainPPO(explore, impala):

    print("Training PPO")
    if impala: print("Using IMPALA CNN")
    if explore: print("Using ExploreGo")

    if impala:
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=1
            )
        )

    else:
        policy_kwargs = {}
    
    env = make_env(train=True, algo='ppo')
    model = PPO(
        "CnnPolicy",
        n_steps=256,
        env=env,
        policy_kwargs=policy_kwargs,
        batch_size=2048,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=linear_schedule(1.75e-4),
        n_epochs=3,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tensorboard_log
        )
    
    model.learn(total_timesteps=100e6)

    model_name ="coinrun-ppo"

    if impala:
        model_name += "-impala"

    if explore:
        model_name += "-explore"

    model.save(model_name)


def trainDQN(explore, impala):

    if impala:
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=1
            )
        )

    else:
        policy_kwargs = {}

    env = make_env(train=True, algo='dqn')
    model = DQN(
        "CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(5e-5),
        batch_size=512,
        gamma=0.99,
        buffer_size=400000,
        target_update_interval=64000,
        learning_starts=250000,
        exploration_final_eps=0.025,
        exploration_initial_eps=1.0,
        max_grad_norm=10,
        train_freq=1,
        exploration_fraction=0.1,
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    model.learn(total_timesteps=100e6)
    model.save("coinrun-dqn")



def trainRainbow(explore, impala):
    
    env = make_env(train=True, algo='rdqn')

    if impala:
        policy_kwargs = dict(
            n_quantiles=200,
            net_arch=[256, 256],
            dueling=True,
            noisy=False,
            noisy_kwargs={
                'sigma': 0.2
            },
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=512,
                depths=[16, 32, 32],
                scale=1
            )
        )

    else:
        policy_kwargs = dict(
            n_quantiles=200,
            net_arch=[256, 256],
            dueling=True,
            noisy=False,
            noisy_kwargs={
                'sigma': 0.2
                }
            )


    model = Rainbow(
        RainbowPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(5e-5),
        batch_size=512,
        gamma=0.99,
        buffer_size=400000,
        target_update_interval=64000,
        learning_starts=250000,
        exploration_final_eps=0.025,
        exploration_initial_eps=1.0,
        max_grad_norm=10,
        train_freq=1,
        exploration_fraction=0.1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    model.learn(total_timesteps=100e6)
    model.save("coinrun-rainbow")


def parser():

    parser = argparse.ArgumentParser(description="Entrenar modelos en CoinRun")
    parser.add_argument('--algo', type=str, choices=['ppo', 'dqn', 'rdqn'], required=True, help='Algoritmo a entrenar')
    parser.add_argument('--explore', action='store_true', help='Usar ExploreGo')
    parser.add_argument('--impala', action='store_true', help='Usar Impala CNN como extractor de caracteristicas')

    return parser.parse_args()


if __name__ == "__main__":

    args = parser()

    if args.algo == 'ppo':

        trainPPO(args.explore, args.impala)

    elif args.algo == 'dqn':
        
        trainDQN(args.explore, args.impala)

    elif args.algo == 'rdqn':

        trainRainbow(args.explore, args.impala)