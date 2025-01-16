import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DQN, PPO, SAC

from CNN2048 import CNN2048

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "CNN_k3n64_s1_d64-extra_delta_RW-illegal_10_10-empty_log_d10-frac_05-taUp_2-lr_e3",

    "algorithm": DQN,
    "algorithm_name": "DQN",
    "policy_network": "MlpPolicy",
    "save_path": "models",

    "epoch_num":500,
    "timesteps_per_epoch": 10000,
    "eval_episode_num": 10,
    "learning_rate": 1e-3,
    "exploration_fraction": 0.5,
    "target_update_interval": 20000,

}


def make_env():
    env = gym.make('2048-v0')
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        avg_highest += info[0]['highest']
        avg_score   += info[0]['score']

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num
        
    return avg_score, avg_highest

def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
    for epoch in range(config["epoch_num"]):

        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        ### Evaluation
        # print(config["run_id"])
        # print("Epoch: ", epoch)
        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        
        # print("Avg_score:  ", avg_score)
        # print("Avg_highest:", avg_highest)
        # print()
        # wandb.log(
        #     {"avg_highest": avg_highest,
        #      "avg_score": avg_score}
        # )
        

        ### Save best model
        if current_best < avg_score:
            print("Saving Model")
            current_best = avg_score
            save_path = config["save_path"]
            RL_algo = config["algorithm_name"]
            steps_epoch = config["timesteps_per_epoch"]
            settings = config["run_id"]
            model.save(f"{save_path}/{RL_algo}/{steps_epoch}-{settings}-{epoch}")

        # print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="RL_hw3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    # Create training environment 
    num_train_envs = 2
    train_env = DummyVecEnv([make_env for _ in range(num_train_envs)])

    # Create evaluation environment 
    eval_env = DummyVecEnv([make_env])  

    # custom CNN model
    policy_kwargs = dict(
        features_extractor_class=CNN2048,
        features_extractor_kwargs=dict(features_dim=64),
    )

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages

    model = my_config["algorithm"](
        my_config["policy_network"], 
        train_env, 
        verbose=2,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
        exploration_fraction=my_config["exploration_fraction"],
        target_update_interval=my_config["target_update_interval"],
        policy_kwargs=policy_kwargs,
    )
    # print(model.policy)
    train(eval_env, model, my_config)