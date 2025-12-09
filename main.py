import numpy as np
import matplotlib 
matplotlib.use('TkAgg') #Amirreza
import matplotlib.pyplot as plt
from datetime import datetime
import sys

import warnings  # I dont make mistakes
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-scatter'")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

from train import train_DQN_Agent, train_PG_Agent
from evaluate import eval_heuristics_dqn, eval_heuristics_pg
from logger import Logger
from utils.plotting import plot_series




if __name__ == "__main__":

    # Create log file                                                                                                                                    
    logfilename = "log.txt"                                                                                                                              
    if len(sys.argv) > 1:                                                         
        logfilename = sys.argv[1]                                                                                                                        
    datetime_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")                                                                                         
    logfilepath = "log/" + datetime_string + "_" + logfilename

    # Create log
    #log = Logger(logfilepath)
    log = Logger(logfilename)

    problem_type = "milp"

    n_train_ep = 100
    n_train_instances = 2
    n_train_trajs = 2
    train_cutoff = 100
    
    n_test_ep = 50
    #plotting settings
    ma_window = 50

    # Policy Gradient Agent
    print("\n\nTraining PG Agent...\n")
    rl_agent, PG_rewards, PG_nodes = train_PG_Agent(log,
                                              num_episodes=n_train_ep,
                                              num_instances=n_train_instances,
                                              num_trajs=n_train_trajs,
                                              problem_type=problem_type,
                                              cutoff=train_cutoff)
    eval_results = eval_heuristics_pg(log, agent=rl_agent, problem_type=problem_type, num_test=n_test_ep)

    plot_series(PG_rewards, "Reward per episode: Policy Gradient", "Reward", MA_window=ma_window, mode="lines")
    plot_series(PG_nodes, "Nodes explored per episode: Policy Gradient", "Nodes explored", MA_window=ma_window , mode="lines")
    #DQN Agent
    print("\n\nTraining DQN Agent...\n")
    DQN_rl_agent, DQN_rewards, DQN_nodes = train_DQN_Agent(log,num_episodes=n_train_ep,
                                                           problem_type=problem_type,cutoff=train_cutoff)
    DQN_eval_results = eval_heuristics_dqn(log, agent=DQN_rl_agent, problem_type=problem_type, num_test=n_test_ep)
    plot_series(DQN_rewards, "Reward per episode: DQN", "Reward", MA_window=ma_window)
    plot_series(DQN_nodes, "Nodes explored per episode: DQN", "Nodes explored", MA_window=ma_window)
