import wandb
import numpy as np

epochs = 50
lr = 0.01

if __name__ == "__main__":
    # login
    wandb.login()
    run = wandb.init(
    # Set the project where this run will be logged
        project="rl-hw2",
        # Track hyperparameters and run metadata
    )
    bias_mc = np.load("./data/bias_mc.npy")
    bias_td = np.load("./data/bias_td.npy")
    var_mc = np.load("./data/var_mc.npy")
    var_td = np.load("./data/var_td.npy")
    # plot bias_mc, bias_td, var_mc, var_td with wandb


    # wandb.log({"Array Plot": wandb.plot.line_series(
    #     xs=np.arange(len(bias_mc)),  # X-axis values (indices of the array)
    #     ys=[bias_mc, bias_td],                # Y-axis values (the array itself)
    #     keys=["bias_mc", "bias_td"],         # Label for the plot
    #     title="Bias : MC vs TD",  # Title of the plot
    #     xname="states",            # Label for the x-axis
    # )})
    
    wandb.log({"Array Plot": wandb.plot.line_series(
        xs=np.arange(len(var_mc)),  # X-axis values (indices of the array)
        ys=[var_mc, var_td],                # Y-axis values (the array itself)
        keys=["var_mc", "var_td"],         # Label for the plot
        title="Variance : MC vs TD",  # Title of the plot
        xname="states",            # Label for the x-axis
    )})
