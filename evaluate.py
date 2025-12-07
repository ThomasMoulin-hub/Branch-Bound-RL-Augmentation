import numpy as np
from instances_generator import get_generator
from gnn import milp_to_pyg_data
from bnb_env import BBEnv
from agents.agent_factory import name_act

def eval_heuristics_dqn(log, agent=None, problem_type="milp", cutoff=1000, num_test = 150):
    
    gen = get_generator(problem_type)

    results = { # all our heuristics, change structure when ill update
        "BestFirst": [],
        "DFS": [],
        "WorstFirst": [],
        "BFS": [],
        "BestEstimate": [],
        "RL": []
    }
    log.joint(f"\nEvaluating {len(results)} heuristics on {num_test} test instances...\n")


    orig_eps = None
    if agent is not None:
        orig_eps = agent.epsilon
        agent.epsilon = 0.0
    else:
        raise Exception("agent is None")


    for _ in range(num_test):
        instance = gen.generate() # new instance for each test
        pyg_data = milp_to_pyg_data(instance)


        env = BBEnv(instance)
        gs = env.get_state_features()
        while not env.done and env.steps < cutoff:
            action = agent.get_action(pyg_data, gs, greedy=True)
            gs, _, _ = env.step(action)
        results["RL"].append(env.steps)


        for name, action in name_act:
            env = BBEnv(instance)
            while not env.done and env.steps < cutoff:
                env.step(action)
            results[name].append(env.steps)

    if agent is not None and orig_eps is not None:
        agent.epsilon = orig_eps

    log.joint("\n--- Average results (number of nodes explored) ---\n")
    for k, v in results.items():
        if v:
            log.joint(f"{k}: {np.mean(v):.2f} ± {np.std(v):.2f}\n")
        else:
            log.joint(f"{k}: no data\n")

    return results


# Literally the above but without epsilon, I didn't want to check cases
def eval_heuristics_pg(log, agent=None, problem_type="milp", cutoff=1000, num_test = 150):
    
    gen = get_generator(problem_type)

    results = { # all our heuristics, change structure when ill update
        "BestFirst": [],
        "DFS": [],
        "WorstFirst": [],
        "BFS": [],
        "BestEstimate": [],
        "RL": []
    }
    log.joint(f"\nEvaluating {len(results)} heuristics on {num_test} test instances...\n")


    for _ in range(num_test):
        instance = gen.generate() # new instance for each test
        pyg_data = milp_to_pyg_data(instance)


        env = BBEnv(instance)
        gs = env.get_state_features()
        while not env.done and env.steps < cutoff:
            action = agent.get_action(pyg_data, gs)
            gs, _, _ = env.step(action)
        results["RL"].append(env.steps)


        for name, action in name_act:
            env = BBEnv(instance)
            while not env.done and env.steps < cutoff:
                env.step(action)
            results[name].append(env.steps)

    log.joint("\n--- Average results (number of nodes explored) ---\n")
    for k, v in results.items():
        if v:
            log.joint(f"{k}: {np.mean(v):.2f} ± {np.std(v):.2f}\n")
        else:
            log.joint(f"{k}: no data\n")

    return results