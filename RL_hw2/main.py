import random
import numpy as np
import json
import time

from algorithms import (
    MonteCarloPrediction,
    TDPrediction,
    NstepTDPrediction,
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 2-1
STEP_REWARD     = -0.1
GOAL_REWARD     = 1.0
TRAP_REWARD     = -1.0
INIT_POS        = [0]
DISCOUNT_FACTOR = 0.9
POLICY          = None
MAX_EPISODE     = 300
LEARNING_RATE   = 0.01
NUM_STEP        = 3
# 2-2
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500

def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt", init_pos: list = None):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=init_pos,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_MC_prediction(grid_world: GridWorld,seed):
    print(f"Run MC prediction. Seed:{seed}")
    prediction = MonteCarloPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"Monte Carlo Prediction",
        show=False,
        filename=f"MC_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_TD_prediction(grid_world: GridWorld, seed):
    print(f"Run TD(0) prediction. Seed:{seed}")
    prediction = TDPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        learning_rate=LEARNING_RATE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"TD(0) Prediction",
        show=False,
        filename=f"TD0_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_NstepTD_prediction(grid_world: GridWorld,seed):
    print(f"Run N-step TD prediction. Seed:{seed}")
    prediction = NstepTDPrediction(
        grid_world,
        learning_rate=LEARNING_RATE,
        num_step=NUM_STEP,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed=seed,
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"N-step TD Prediction",
        show=False,
        filename=f"NstepTD_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()

def run_MC_policy_iteration(grid_world: GridWorld, iter_num: int):
    print(bold(underline("MC Policy Iteration")))
    policy_iteration = MonteCarloPolicyIteration(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"MC Policy Iteration",
        show=False,
        filename=f"MC_policy_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

def run_SARSA(grid_world: GridWorld, iter_num: int):
    print(bold(underline("SARSA Policy Iteration")))
    policy_iteration = SARSA(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"SARSA",
        show=False,
        filename=f"SARSA_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_Q_Learning(grid_world: GridWorld, iter_num: int):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"Q_Learning",
        show=False,
        filename=f"Q_Learning_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

if __name__ == "__main__":
    time_c = time.time()

    ## report
    # grid_world = init_grid_world("maze.txt",INIT_POS)
    # V_GT = np.load("./sample_solutions/prediction_GT.npy")
    # state_num = grid_world.get_state_space()
    # run = 50
    # V_avg_mc = 0
    # V_avg_td = 0
    # V_i_mc = np.zeros((50, state_num))
    # V_i_td = np.zeros((50, state_num))
    # var_mc = np.zeros(state_num)
    # var_td = np.zeros(state_num)

    # # bias
    # for seed in range(run):
    #     V_i_mc[seed] = run_MC_prediction(grid_world, seed)
    #     V_i_td[seed] = run_TD_prediction(grid_world, seed)
    #     V_avg_mc += 1/run * V_i_mc[seed] 
    #     V_avg_td += 1/run * V_i_td[seed] 
    # bias_mc = V_avg_mc - V_GT
    # bias_td = V_avg_td - V_GT

    # # variance
    # for i in range(state_num):
    #     for j in range(run):
    #         var_mc[i] += 1/run * (V_i_mc[j][i] - V_avg_mc[i])**2
    #         var_td[i] += 1/run * (V_i_td[j][i] - V_avg_td[i])**2

    # print("bias mc : ", bias_mc)
    # print("bias td : ", bias_td)
    # print("var mc : ", var_mc)
    # print("var td : ", var_td)

    # # save data
    # np.save("./data/bias_mc.npy", bias_mc)
    # np.save("./data/bias_td.npy", bias_td)
    # np.save("./data/var_mc.npy", var_mc)
    # np.save("./data/var_td.npy", var_td)

    ## 2-1

    seed = 1
    grid_world = init_grid_world("maze.txt", INIT_POS)
    run_MC_prediction(grid_world, seed)
    run_TD_prediction(grid_world,seed)
    run_NstepTD_prediction(grid_world,seed)

    ## 2-2
    grid_world = init_grid_world("maze.txt")
    run_MC_policy_iteration(grid_world, 512000) # sometimes it stucks
    run_SARSA(grid_world, 512000)
    run_Q_Learning(grid_world, 50000)

    time_elaped = time.time() - time_c
    print(f"Time Elapsed: {time_elaped} sec")