import numpy as np
import matplotlib 
matplotlib.use('TkAgg') #Amirreza
import matplotlib.pyplot as plt

import warnings  # I dont make mistakes
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-scatter'")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")


from instances_generator import SetCoverGenerator, RandomMILPGenerator
from gnn import milp_to_pyg_data
from bnb_env import BBEnv
from dqn_agent import DQNAgent



cutoff = 1000

name_act = [ # make this cleaner with above also
    ("BestFirst", 0),
    ("DFS", 1),
    ("WorstFirst", 2),
    ("BFS", 3),
    ("BestEstimate", 4),
]
n_heuristics = len(name_act)


problems = {
    "milp": {"n_cons": 6, "n_var": 32, "density": 0.4}
}


def get_generator(problem_type: str):
    match problem_type:
        case "milp":
            p = problems["milp"] # problem parameters, easier to change and personalize
            return RandomMILPGenerator(n_cons=p["n_cons"], n_vars=p["n_var"], density=p["density"])
        case _:
            raise Exception("pls provide existing problem")


def get_agent(problem_type: str):
    match problem_type:
        case "milp":
            p = problems["milp"]
            return DQNAgent(input_dim=p["n_cons"], hidden_dim=32, action_dim=n_heuristics, global_dim=10)
        case _:
            raise Exception("pls pick existing problem")



def train_rl_bnb(num_episodes, problem_type="milp", batch_size=32):
    gen = get_generator(problem_type)
    agent = get_agent(problem_type)

    rewards_history = []  # batch avg
    nodes_history = []    # batch avg

    print(f"Training on {num_episodes} episodes ({problem_type}):")

    batch_reward = 0.0
    batch_nodes = 0.0
    batch_loss = 0.0
    batch_count = 0

    for episode in range(num_episodes):

        instance = gen.generate()
        env = BBEnv(instance)

        pyg_data = milp_to_pyg_data(instance)
        global_state = env.get_state_features()

        total_reward = 0.0
        buffer = []
        truncated = False

        while not env.done:
            action = agent.get_action(pyg_data, global_state)
            next_global_state, reward, done = env.step(action)

            buffer.append(
                (pyg_data, global_state, action, reward,
                 pyg_data, next_global_state, done)
            )

            global_state = next_global_state
            total_reward += reward

            if env.steps > cutoff:
                truncated = True
                break

        if truncated and buffer:
            pyg, glob, act, rew, next_pyg, next_glob, _ = buffer[-1]
            buffer[-1] = (pyg, glob, act, rew, next_pyg, next_glob, True)

        loss = agent.update(buffer)

        # accumulate batch stats
        batch_reward += total_reward
        batch_nodes += env.steps
        batch_loss += loss
        batch_count += 1

        # when batch is full, log averages
        if (episode + 1) % batch_size == 0:
            avg_reward = batch_reward / batch_count
            avg_nodes = batch_nodes / batch_count
            avg_loss = batch_loss / batch_count

            rewards_history.append(avg_reward)
            nodes_history.append(avg_nodes)

            print(
                f"Episode {episode + 1}: "
                f"batch_avg_reward = {avg_reward:.2f}, "
                f"batch_avg_nodes = {avg_nodes:.2f}, "
                f"batch_avg_loss = {avg_loss:.4f}, "
                f"eps = {agent.epsilon:.3f}"
            )

            batch_reward = 0.0
            batch_nodes = 0.0
            batch_loss = 0.0
            batch_count = 0

    return agent, rewards_history, nodes_history




def eval_heuristics(agent=None, problem_type="milp", num_test = 150):
    
    gen = get_generator(problem_type)

    results = { # all our heuristics, change structure when ill update
        "BestFirst": [],
        "DFS": [],
        "WorstFirst": [],
        "BFS": [],
        "BestEstimate": [],
        "RL": []
    }
    print(f"\nEvaluating {len(results)} heuristics on {num_test} test instances...")


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

    print("\n--- Average results (number of nodes explored) ---")
    for k, v in results.items():
        if v:
            print(f"{k}: {np.mean(v):.2f} ± {np.std(v):.2f}")
        else:
            print(f"{k}: no data")

    return results




if __name__ == "__main__":
    n_train_ep = 512
    n_test_ep = 100
    problem_type = "milp"
    batch_size = 16

    rl_agent, rewards, nodes = train_rl_bnb(
        num_episodes=n_train_ep,
        problem_type=problem_type,
        batch_size=batch_size,
    )
    eval_results = eval_heuristics(agent=rl_agent, problem_type=problem_type, num_test=n_test_ep)

    x = np.arange(len(rewards)) * batch_size  # approx position in episodes

    plt.figure()
    plt.plot(x, rewards, label=f"batch_avg_reward (batch={batch_size})")
    plt.title("Reward per batch")
    plt.xlabel("Episode")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x, nodes, label=f"batch_avg_nodes (batch={batch_size})")
    plt.title("Nodes explored per batch")
    plt.xlabel("Episode")
    plt.grid(True)
    plt.legend()
    plt.show()
