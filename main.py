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


    rl_agent, rewards, nodes = train_PG_Agent(log,
                                              num_episodes=n_train_ep,
                                              num_instances=n_train_instances,
                                              num_trajs=n_train_trajs,
                                              problem_type=problem_type,
                                              cutoff=train_cutoff)
    eval_results = eval_heuristics_pg(log, agent=rl_agent, problem_type=problem_type, num_test=n_test_ep)

    # put this in util or plot file idk
    plot_series(rewards, "Reward per episode", "Reward", w=10)
    plot_series(nodes, "Nodes explored per episode", "Nodes explored", w=10)