import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN

import numpy as np
from collections import Counter

register(
    id='2048-eval',
    entry_point='envs:Eval2048Env'
)

def evaluation(env, model, render_last, eval_num=100):
    """We only evaluate seeds 0-99 as our public test cases."""
    score = []
    highest = []

    ### Run eval_num times rollouts,
    for seed in range(eval_num):
        done = False
        # Set seed and reset env using Gymnasium API
        obs, info = env.reset(seed=seed)

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            # env.render()

        # Render the last board state of each episode
        # print("Last board state:")
        # env.render()

        score.append(info['score'])
        highest.append(info['highest'])

    ### Render last rollout
    if render_last:
        print("Rendering last rollout")
        done = False
        obs, info = env.reset(seed=eval_num-1)
        env.render()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            env.render()

        
    return score, highest


if __name__ == "__main__":
    # model_path = "models/sample_model/21"  # Change path name to load different models
    # model_path = "models/DQN/10000-CNN_k3n64_s1_d64-extra_delta_RW-illegal_10_10-empty_log_d10-frac_05-taUp_2-lr_e3-n500_2-435"  # Change path name to load different models
    # model_path = "models/PPO/10000-PPO-extra_delta-illegal_10_1-empty_log_d10-lr_e3-25"  # Change path name to load different models
    # model_path = "models/best/DQN/10000-CNN_k3n64_s1_d64-extra_delta_RW-illegal_10_10-empty_log_d10-frac_05-taUp_2-lr_e3-n500_2-435"  # Change path name to load different models
    model_path = "model"  # best so far
    env = gym.make('2048-eval')

    ### Load model with SB3
    # Note: Model can be loaded with arbitrary algorithm class for evaluation
    # (You don't necessarily need to use PPO for training)
    # model = PPO.load(model_path)
    model = DQN.load(model_path)
    
    eval_num = 100
    score, highest = evaluation(env, model, True, eval_num)

    print("Avg_score:  ", np.sum(score)/eval_num)
    print("Avg_highest:", np.sum(highest)/eval_num)


    # bar_x = []
    # bar_y = []
    print(f"Counts: (Total of {eval_num} rollouts)")
    c = Counter(highest)
    for item in (sorted(c.items(),key = lambda i: i[0])):
        print(f"{item[0]}: {item[1]}")
        # bar_x.append(item[0].item())
        # bar_y.append(item[1])
    
    # import matplotlib.pyplot as plt

    # print(bar_x)
    # print(bar_y)
    # x = np.arange(len(bar_x))
    # plt.bar(x, bar_y)
    # plt.xticks(x, bar_x)
    # plt.xlabel('Best tile number')
    # plt.ylabel('Count')
    # plt.title('Distribution of 2048 best tile')
    # plt.show()