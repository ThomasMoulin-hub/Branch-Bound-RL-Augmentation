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
from evaluate import eval_heuristics_dqn, eval_heuristics_pg, eval_heuristics_both
from logger import Logger
from utils.plotting import plot_series
from agents.agent_factory import get_agent




if __name__ == "__main__":

    # Create log file                                                                                                                                    
    logfilename = "log.txt"                                                                                                                              
    if len(sys.argv) > 1:                                                         
        logfilename = sys.argv[1]                                                                                                                        
    datetime_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")                                                                                         
    logfilepath = "log/" + datetime_string + "_" + logfilename

    # Create log
    log = Logger(logfilepath)
    #log = Logger(logfilename)

    problem_type = "milp"

    n_train_ep = 250
    n_train_instances = 2
    n_train_trajs = 2

    # Need more episodes for DQN to see same number of trajectories as PG
    n_train_ep_DQN = n_train_ep * n_train_instances * n_train_trajs
    
    # Number of tests
    n_test_ep = 75

    # plotting settings
    ma_window = 25

    # Algorithm settings
    lr = 1e-3
    gamma = 0.95
    cutoff = 250

    # Log settings
    log.joint("Run Settings:\n")
    log.joint(f"n_train_ep {n_train_ep}\n")
    log.joint(f"n_train_instances (pg only) {n_train_instances}\n")
    log.joint(f"n_train_trajs (pg only) {n_train_trajs}\n")
    log.joint(f"cutoff (pg only) {cutoff}\n")
    log.joint(f"lr {lr}\n")
    log.joint(f"gamma {gamma}\n")


    # Policy Gradient Agent
    print("\n\nTraining PG Agent...\n")
    PG_agent, PG_rewards, PG_nodes = train_PG_Agent(log, num_episodes=n_train_ep,
                                                    num_instances=n_train_instances,
                                                    num_trajs=n_train_trajs,
                                                    problem_type=problem_type,
                                                    cutoff=cutoff,
                                                    lr=lr,
                                                    gamma=gamma)
    plot_series(PG_rewards, "Reward per trajectory: Policy Gradient", "Reward", MA_window=ma_window, mode="lines", filename="PG_rew.png")
    plot_series(PG_nodes, "Nodes explored per trajectory: Policy Gradient", "Nodes explored", MA_window=ma_window , mode="lines", filename="PG_nodes.png")
    
    
    #DQN Agent
    print("\n\nTraining DQN Agent...\n")
    DQN_agent, DQN_rewards, DQN_nodes = train_DQN_Agent(log,num_episodes=n_train_ep_DQN,
                                                        problem_type=problem_type,
                                                        cutoff=cutoff,
                                                        lr=lr,
                                                        gamma=gamma)
    plot_series(DQN_rewards, "Reward per trajectory: DQN", "Reward", MA_window=ma_window, filename="DQN_rew.png")
    plot_series(DQN_nodes, "Nodes explored per trajectory: DQN", "Nodes explored", MA_window=ma_window, filename="DQN_nodes.png")
    
    DQN_agent = get_agent(problem_type, "dqn", lr, gamma)
    PG_agent = get_agent(problem_type, "pg", lr, gamma)
    eval_results = eval_heuristics_both(log, agent_dqn=DQN_agent, agent_pg=PG_agent,
                                        problem_type=problem_type, num_test=n_test_ep, cutoff=cutoff)

