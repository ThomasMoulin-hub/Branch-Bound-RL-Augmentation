import numpy as np
from bnb_env import BBEnv
from instances_generator import get_generator
from agents.agent_factory import get_agent
from gnn import milp_to_pyg_data


def discounted_rewards(r, gamma):
    """ HELPER: take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


def train_DQN_Agent(log, num_episodes, problem_type="milp", cutoff=1000):
    """ Train DQN agent using one instance per epsiode """
    gen = get_generator(problem_type)
    agent = get_agent(problem_type, "dqn")

    rewards_history = []
    nodes_history = []

    log.joint(f"Training on {num_episodes} episodes ({problem_type}):\n")

    for episode in range(num_episodes):

        instance = gen.generate() # get a new instance
        env = BBEnv(instance)

        pyg_data = milp_to_pyg_data(instance)   # Static graph representation of the MILP (no change within episode)
        global_state = env.get_state_features() # not static

        total_reward = 0.0
        buffer = []          # stores transitions for the episode
        truncated = False    # step cutoff

        while not env.done:
            
            action = agent.get_action(pyg_data, global_state)

            next_global_state, reward, done = env.step(action)

            buffer.append( # Store transition
                (pyg_data, global_state, action, reward,
                 pyg_data, next_global_state, done))

            global_state = next_global_state
            total_reward += reward


            if env.steps > cutoff: # safety cutoff on the nb of explored nodes
                truncated = True
                break


        # If the episode was truncated we mark last transition as terminal for training
        if truncated and buffer: 
            pyg, glob, act, rew, next_pyg, next_glob, _ = buffer[-1]
            buffer[-1] = (pyg, glob, act, rew, next_pyg, next_glob, True)


        # Single update on all collected transitions
        loss = agent.update(buffer)
        rewards_history.append(total_reward)
        nodes_history.append(env.steps)
        if episode % 10 == 0:
            log.joint(
                f"Episode {episode}: R = {total_reward:.2f}, "
                f"Nodes = {env.steps}, eps = {agent.epsilon:.3f}, loss = {loss:.4f}\n"
            )

    return agent, rewards_history, nodes_history


def train_PG_Agent(log, num_episodes, num_instances, num_trajs, problem_type="milp", cutoff=1000):
    gen = get_generator(problem_type)
    agent = get_agent(problem_type, "pg")  # TODO: set to PG agent

    rewards_history = []
    nodes_history = []

    batch_size = num_instances * num_trajs

    log.joint(f"Training on {num_episodes} episodes ({problem_type}):\n")

    for episode in range(num_episodes):
        # For recording trajectories from current policy
        OBS = []
        ACTS = []
        VAL = []

        for _instance in range(num_instances):
            # Generate instance
            instance = gen.generate()
            env = BBEnv(instance)
            pyg_data = milp_to_pyg_data(instance)  # Static graph representation of the MILP (no change within episode)

            for _traj in range(num_trajs):
                # To keep a record of states actions and reward for each episode
                obss = []  # states (pyg_data, global_state)
                acts = []  # actions
                rews = []  # instant rewards

                # Reset environment
                env.reset()
                global_state = env.get_state_features() # not static
                truncated = False    # step cutoff
                done = False

                # Compute one trajectory using the current policy
                while not done:
                    obss.append((pyg_data, global_state))

                    # Take action and observe reward
                    action = agent.get_action(pyg_data, global_state)
                    next_global_state, reward, done = env.step(action)

                    # Update environment
                    global_state = next_global_state

                    acts.append(action)
                    rews.append(reward)

                    if env.steps > cutoff: # safety cutoff on the nb of explored nodes
                        truncated = True
                        break

                # Use discounted_rewards function to compute hatVs using instant rewards in rews
                # Record the computed hatVs in VAL, states obss in OBS, and actions acts in ACTS, for batch update
                hatV = discounted_rewards(rews, agent.gamma)
                OBS.append(obss)
                ACTS.extend(acts)
                VAL.extend(hatV)

                rewards_history.append(np.sum(rews))
                nodes_history.append(env.steps)
        
        # After collecting num_instances * num_trajs trajectories...
        ACTS_np = np.array(ACTS)
        VAL_np = np.array(VAL)

        # Single update on all collected transitions
        loss = agent.update(OBS, ACTS_np, VAL_np)

        if episode % 10 == 0:
            # recent averages
            avg_rew = np.sum(rewards_history[-batch_size:]) / batch_size
            avg_nodes = np.sum(nodes_history[-batch_size:]) / batch_size
            log.joint(
                f"Episode {episode}: R = {avg_rew:.2f}, "
                f"Nodes (avg) = {avg_nodes:.2f}, loss = {loss:.4f}\n"
            )

    return agent, rewards_history, nodes_history
        