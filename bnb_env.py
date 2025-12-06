import numpy as np
import scipy.optimize as opt
import random


class BBNode:
    def __init__(self, lower_bound, fixed_vars, depth):
        self.lower_bound = lower_bound
        self.fixed_vars = fixed_vars
        self.depth = depth
        self.lp_solution = None  # relaxed solution


class BBEnv:
    """
    Branch-and-Bound environment for (cover/packing) MILPs.

    - State: global B&B statistics (depth, bounds, fractionality, etc.)
    - Action: node-selection heuristic.
        0: best-bound (min lower_bound)
        1: depth-first (max depth)
        2: worst-bound (max lower_bound)
        3: breadth-first (min depth)
        4: best-estimate (approx)
        >=5: random node
    """

    def __init__(self, instance):
        self.instance = instance
        self.A = instance['A']
        self.c = instance['c']
        self.b = instance['b']

        self.problem_type = instance.get('type', 'cover')
        self.n_vars = len(self.c)

        # root node: no fixed vars
        root = BBNode(
            lower_bound=-np.inf,
            fixed_vars=np.array([None] * self.n_vars),
            depth=0
        )

        self.global_ub = float('inf')  # best known feasible solution value
        self.fringe = [root]
        self.steps = 0
        self.done = False

        # Pre-solve LP at root
        lb, x = self.process_node_lp(root)
        if x is not None and self.is_integer(x):
            self.global_ub = lb
            self.done = True

    def solve_lp(self, fixed_vars):
        """
        Solve the LP relaxation for the current MILP with some variables fixed.

        fixed_vars: array of length n_vars, each entry in {0,1,None}.
        None means free, otherwise that var is fixed to 0 or 1.
        """
        bounds = []
        for v in fixed_vars:
            if v is None:
                bounds.append((0, 1))
            else:
                bounds.append((v, v))

        if self.problem_type == 'cover':
            # Set cover: Ax >= b -> rewrite as -Ax <= -b for linprog
            A_ub = -self.A
            b_ub = -self.b
        else:
            # Packing: Ax <= b
            A_ub = self.A
            b_ub = self.b

        res = opt.linprog(
            self.c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs'
        )

        if res.success:
            return res.fun, res.x
        else:
            # LP infeasible
            return float('inf'), None

    def process_node_lp(self, node: BBNode):
        """
        Run the LP relaxation at this node and store (lower_bound, lp_solution).
        """
        lb, x = self.solve_lp(node.fixed_vars)
        node.lower_bound = lb
        node.lp_solution = x
        return lb, x

    @staticmethod
    def is_integer(x):
        """
        Check if a vector is (numerically) binary 0/1.
        """
        if x is None:
            return False
        return np.allclose(x, np.round(x), atol=1e-5)

    @staticmethod
    def node_fractionality(node: BBNode) -> float:
        """
        Average fractionality of the LP solution at this node.
        0 = almost integral, 1 = very fractional.
        """
        x = node.lp_solution
        if x is None:
            return 1.0
        frac = np.abs(x - np.round(x))
        return float(np.mean(frac))

    def node_estimate(self, node: BBNode) -> float:
        """
        Approximate 'best-estimate' for MINIMIZATION.

        est = lb + frac * gap  (if UB available)
            = lb + frac        (otherwise)

        where:
          - lb   = node.lower_bound
          - frac = average fractionality of the LP solution
          - gap  = global_ub - lb  (>= 0)
        """
        lb = node.lower_bound
        frac = self.node_fractionality(node)

        if self.global_ub < float('inf'):
            gap = max(0.0, self.global_ub - lb)
            return lb + frac * gap
        else:
            return lb + frac

    def get_state_features(self):
        """
        Rich global B&B features.

        Features:
        0  avg_depth
        1  max_depth
        2  fringe_size
        3  best_lb
        4  worst_lb
        5  mean_lb
        6  std_lb
        7  gap (UB vs best LB)
        8  mean_node_fractionality
        9  max_node_fractionality
        """
        if not self.fringe:
            return np.zeros(10, dtype=np.float32)

        depths = np.array([n.depth for n in self.fringe], dtype=np.float32)
        lbs = np.array([n.lower_bound for n in self.fringe], dtype=np.float32)

        avg_depth = float(np.mean(depths))
        max_depth = float(np.max(depths))
        fringe_size = float(len(self.fringe))

        best_lb = float(np.min(lbs))
        worst_lb = float(np.max(lbs))
        mean_lb = float(np.mean(lbs))
        std_lb = float(np.std(lbs))

        gap = 0.0
        if self.global_ub < float('inf'):
            denom = max(1e-9, abs(best_lb))
            gap = float(abs(self.global_ub - best_lb) / denom)

        # Fractionality stats over fringe
        node_fracs = []
        for n in self.fringe:
            x = n.lp_solution
            if x is None:
                continue
            frac = np.abs(x - np.round(x))
            node_fracs.append(np.mean(frac))

        if node_fracs:
            mean_node_frac = float(np.mean(node_fracs))
            max_node_frac = float(np.max(node_fracs))
        else:
            mean_node_frac = 0.0
            max_node_frac = 0.0

        return np.array([
            avg_depth, max_depth, fringe_size,
            best_lb, worst_lb, mean_lb, std_lb,
            gap, mean_node_frac, max_node_frac
        ], dtype=np.float32)

    def branch(self, node: BBNode):
        """
        Branch on the most fractional variable in node.lp_solution.
        Returns a list of child nodes.
        """
        x = node.lp_solution
        if x is None:
            return []

        frac = np.abs(x - np.round(x))
        idx = np.argmax(frac)
        if frac[idx] < 1e-5:
            # Already integral
            return []

        children = []
        for val in [0, 1]:
            fixed_copy = node.fixed_vars.copy()
            fixed_copy[idx] = val
            child = BBNode(
                lower_bound=node.lower_bound,
                fixed_vars=fixed_copy,
                depth=node.depth + 1
            )
            children.append(child)

        return children

    def step(self, action):
        """
        One B&B step controlled by an action:
        - action 0: best-bound (min lower_bound)
        - action 1: depth-first (max depth)
        - action 2: worst-bound (max lower_bound)
        - action 3: breadth-first (min depth)
        - action 4: best-estimate (approx)
        - action >=5: random node

        Returns: (next_state_features, reward, done)
        """

        self.steps += 1
        reward = -0.1  # small penalty per processed node

        if not self.fringe:
            self.done = True
            return self.get_state_features(), reward, True

        # ----- NODE SELECTION -----
        if action == 0:   # Best-bound (best-first): min lower_bound
            self.fringe.sort(key=lambda x: x.lower_bound)
            node = self.fringe.pop(0)

        elif action == 1:  # Depth-first: max depth
            self.fringe.sort(key=lambda x: x.depth, reverse=True)
            node = self.fringe.pop(0)

        elif action == 2:  # Worst-bound: max lower_bound
            self.fringe.sort(key=lambda x: x.lower_bound, reverse=True)
            node = self.fringe.pop(0)

        elif action == 3:  # Breadth-first: min depth
            self.fringe.sort(key=lambda x: x.depth)
            node = self.fringe.pop(0)

        elif action == 4:  # Best-estimate (approx)
            best_idx = 0
            best_est = self.node_estimate(self.fringe[0])
            for i in range(1, len(self.fringe)):
                est_i = self.node_estimate(self.fringe[i])
                if est_i < best_est:
                    best_est = est_i
                    best_idx = i
            node = self.fringe.pop(best_idx)

        else:              # Random node
            idx = random.randint(0, len(self.fringe) - 1)
            node = self.fringe.pop(idx)

        # ----- PROCESS SELECTED NODE -----
        lb, x = self.process_node_lp(node)

        # Infeasible or dominated node
        if lb == float('inf') or lb >= self.global_ub:
            reward += 0.2  # small bonus for pruning
            if not self.fringe:
                self.done = True
                reward += 2.0  # bonus for finishing the tree
            return self.get_state_features(), reward, self.done

        # If LP solution is integral, update UB and give a reward
        if self.is_integer(x):
            if lb < self.global_ub:
                improvement = self.global_ub - lb if self.global_ub < float('inf') else 0.0
                self.global_ub = lb
                reward += 1.0 + 0.1 * improvement
            if not self.fringe:
                self.done = True
                reward += 2.0  # bonus for finishing the tree
            return self.get_state_features(), reward, self.done

        # Otherwise, branch and process children
        children = self.branch(node)
        for child in children:
            lb_child, x_child = self.process_node_lp(child)

            # Infeasible or dominated
            if lb_child == float('inf') or lb_child >= self.global_ub:
                continue

            if self.is_integer(x_child):
                # Child gives an integer solution
                if lb_child < self.global_ub:
                    self.global_ub = lb_child
                    reward += 1.0
            else:
                self.fringe.append(child)

        if not self.fringe:
            self.done = True
            reward += 2.0  # finishing bonus

        return self.get_state_features(), reward, self.done
