from instances_generator import problems
from agents.dqn_agent import DQNAgent
from agents.policygradient_agent import PolicyGradientAgent

name_act = [ # make this cleaner with above also
    ("BestFirst", 0),
    ("DFS", 1),
    ("WorstFirst", 2),
    ("BFS", 3),
    ("BestEstimate", 4),
]
n_heuristics = len(name_act)


def get_agent(problem_type: str, agent_type: str):
    input_dim = None

    match problem_type:
        case "milp":
            p = problems["milp"]
            input_dim = p["n_cons"]
        case _:
            raise Exception("pls pick existing problem")
        
    match agent_type:
        case "dqn":
            return DQNAgent(input_dim=input_dim, hidden_dim=32, action_dim=n_heuristics, global_dim=10)
        case "pg":
            return PolicyGradientAgent(input_dim=input_dim, hidden_dim=32, action_dim=n_heuristics, global_dim=10)
        case _:
            raise Exception("pls pick existing agent")