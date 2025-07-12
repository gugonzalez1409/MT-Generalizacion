import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from typing import Callable
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecMonitor, VecExtractDictObs
from stable_baselines3.common.callbacks import CheckpointCallback
from models.rainbow.rainbow import Rainbow
from models.rainbow.policies import RainbowPolicy
from models.generalization.ImpalaCNN import ImpalaCNN
from sb3_contrib import RecurrentPPO
from models.utils.envs import make_single_env, vectorizedEnv
from procgen import ProcgenEnv

def linear_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:

        return progress_remaining * initial_value

    return func

def make_env():
    env = ProcgenEnv(
        num_envs=200,
        env_name="coinrun",
        num_levels=200,
        start_level=0,
        distribution_mode="extreme"
    )
    env = VecExtractDictObs(env, "rgb")
    env = VecMonitor(env)
    return env


def trainPPO(explore, impala):
    
    env = make_env()
    model = PPO(
        "CnnPolicy",
        n_steps=64,
        env=env,
        batch_size=1024,
        learning_rate=linear_schedule(2.5e-4),
        n_epochs=3,
        ent_coef=0.01,
        verbose=1
        )
    model.learn(total_timesteps=75e6)

def trainRecurrentPPO(explore, impala):
    
    env = make_env()
    model = RecurrentPPO(
        "CnnLstmPolicy",
        n_steps=64,
        env=env,
        batch_size=1024,
        learning_rate=linear_schedule(2.5e-4),
        n_epochs=3,
        ent_coef=0.01,
        verbose=1
    )
    model.learn(total_timesteps=75e6)

def trainDQN(explore, impala):
    env = make_env()
    model = DQN(
        "CnnPolicy",
        env=env,
        learning_rate=linear_schedule(5e-4),
        buffer_size=100000,
        batch_size=64,
        learning_starts=50000,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        verbose=1,
    )

def trainRainbow(explore, impala):
    
    env = make_env()

    if impala:
        policy_kwargs = dict(
            n_quantiles=200,
            net_arch=[256, 256],
            dueling=True,
            noisy=True,
            noisy_kwargs={
                'sigma': 0.5
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
            noisy=True,
            noisy_kwargs={
                'sigma': 0.5
            }
            )


    model = Rainbow(
        RainbowPolicy,
        env = env,
        learning_rate=linear_schedule(2.5e-4),
        learning_starts=50000,
        policy_kwargs=policy_kwargs,
        buffer_size=100000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.2,
        batch_size=64,
        gamma=0.999,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        verbose=1,
    )

    save_freq = max(10e6 // 11, 1)

    #save_callback = CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix='RDQN_checkpoint', verbose=1)
    model.learn(total_timesteps=75e6) # , callback=save_callback


if __name__ == "__main__":
    trainPPO(explore=False, impala=False)
    # trainRecurrentPPO(explore=True, impala=True)
    # trainDQN(explore=True, impala=True)
    # trainRainbow(explore=True, impala=True)