import numpy as np
import matplotlib 
#matplotlib.use('TkAgg') #Amirreza
import matplotlib.pyplot as plt
from datetime import datetime
import sys

import warnings  # I dont make mistakes
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-scatter'")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")

from train import train_DQN_Agent, train_PG_Agent
from evaluate import eval_heuristics_dqn, eval_heuristics_pg
from logger import Logger


if __name__ == "__main__":

    # Create log file                                                                                                                                    
    logfilename = "log.txt"                                                                                                                              
    if len(sys.argv) > 1:                                                         
        logfilename = sys.argv[1]                                                                                                                        
    datetime_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")                                                                                         
    logfilepath = "log/" + datetime_string + "_" + logfilename

    # Create log
    log = Logger(logfilepath)

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
    plt.figure()
    plt.plot(rewards, label="reward per episode")
    w = 10
    cumS = np.cumsum(np.insert(rewards, 0, 0))
    rw_ma = (cumS[w:] - cumS[:-w]) / float(w)
    plt.plot(range(w - 1, len(rewards)), rw_ma, label=f"reward MA({w})")
    plt.title("Reward per episode")
    plt.xlabel("Episode")
    plt.grid(True)
    plt.legend()
    plt.show()



    plt.figure()
    plt.plot(nodes, label="nodes explored")
    w = 10
    cumS = np.cumsum(np.insert(nodes, 0, 0))
    nd_ma = (cumS[w:] - cumS[:-w]) / float(w)
    plt.plot(range(w - 1, len(nodes)), nd_ma, label=f"nodes MA({w})")
    plt.title("Nodes explored per episode")
    plt.xlabel("Episode")
    plt.grid(True)
    plt.legend()
    plt.show()