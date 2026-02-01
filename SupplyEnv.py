# supply_env_a2cV1.py
import numpy as np
import gym
from gym import spaces
from connectivity import generate_connectivity, validate_paths
from fitness import fitness_function
from utils import compute_power_distribution_by_assignment

class SupplyEnv(gym.Env):

    metadata = {"render.modes": []}

    def __init__(self, G_status, G_limits, switches, loads_power, load_levels, horizon=32):
        super().__init__()
        self.G_status    = np.asarray(G_status, dtype=int)
        self.G_limits    = np.asarray(G_limits, dtype=float)
        self.switches    = list(switches)
        self.loads_power = np.asarray(loads_power, dtype=float)
        self.load_levels = np.asarray(load_levels, dtype=int)

        self.num_loads = len(self.loads_power)
        self.horizon   = int(horizon)
        self.t = 0

        self.conn = generate_connectivity(self.G_status, self.switches)
        self.valid_paths = validate_paths(self.conn)
        self.valid_generators = self._get_valid_generators_from_conn()
        self.default_sources  = self._get_default_sources_from_conn()
        self.shortest_len     = self._compute_shortest_path_len_by_gen()

        # Spaces
        self.action_space = spaces.MultiBinary(self.num_loads)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.num_loads,),
                                            dtype=np.float32)
        self.state = np.zeros(self.num_loads, dtype=np.float32)
        self.last_assigned_gens = np.full(self.num_loads, -1, dtype=int)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self._refresh_connectivity()
        self.state = np.zeros(self.num_loads, dtype=np.float32)
        self.last_assigned_gens = np.full(self.num_loads, -1, dtype=int)
        return self.state

    def step(self, action):
        action = np.asarray(action, dtype=int).clip(0, 1)

        self._refresh_connectivity()
      
        mask = self.get_action_mask()
        infeasible_bits_conn = int(((action == 1) & (mask == 0)).sum())
        action = (action & mask).astype(int)

        assigned_gens, dropped_indices = self._priority_assignment(action)
        self.last_assigned_gens = assigned_gens.copy()

        action_real = (assigned_gens > 0).astype(int)

        power_dist = compute_power_distribution_by_assignment(assigned_gens, self.loads_power)
        over = np.maximum(0.0, power_dist - self.G_limits)
        penalty_over = float(over.sum())

        fitness = fitness_function(action_real, self.loads_power, self.load_levels)
        penalty_projection = 0.2 * float(len(dropped_indices))
        penalty_infeasible = 5.0 * float(infeasible_bits_conn)
        reward = fitness - 5.0 * penalty_over - penalty_infeasible - penalty_projection

        self.state = action_real.astype(np.float32)
        self.t += 1
        done = (self.t >= self.horizon)

        info = {
            "fitness_single": float(fitness),
            "reward_single": float(reward),
            "penalty_over": float(penalty_over),
            "penalty_infeasible": float(penalty_infeasible + penalty_projection),
            "power_dist": power_dist,
            "over": over,
            "assigned_gens": assigned_gens,
            "projection_dropped": dropped_indices,
            "t": self.t,
        }
        return self.state, float(reward), bool(done), info

    def get_action_mask(self):
        mask = np.ones(self.num_loads, dtype=int)
        for i, gset in enumerate(self.valid_generators):
            if len(gset) == 0:
                mask[i] = 0
        return mask

    def _refresh_connectivity(self):
        self.conn = generate_connectivity(self.G_status, self.switches)
        self.valid_paths = validate_paths(self.conn)
        self.valid_generators = self._get_valid_generators_from_conn()
        self.default_sources  = self._get_default_sources_from_conn()
        self.shortest_len     = self._compute_shortest_path_len_by_gen()

    def _iter_valid_candidates_from_conn(self, load_idx: int):

        key = f"Load{load_idx+1}"
        for p in self.valid_paths.get(key, []):
            g = int(p.get("g", -1))
            if 1 <= g <= 4 and self.G_status[g-1] == 1:
                yield (g, p.get("bits", []))

    def _get_valid_generators_from_conn(self):
        gens = []
        for i in range(self.num_loads):
            valid = set()
            for g, _bits in self._iter_valid_candidates_from_conn(i):
                valid.add(g)
            gens.append(valid)
        return gens

    def _get_default_sources_from_conn(self):
        sources = []
        for i in range(self.num_loads):
            best_g, best_len = -1, 1e9
            for g, bits in self._iter_valid_candidates_from_conn(i):
                L = len(bits)
                if L < best_len:
                    best_len, best_g = L, g
            sources.append(best_g)
        return np.asarray(sources, dtype=int)

    def _compute_shortest_path_len_by_gen(self):

        shortest_len = [{} for _ in range(self.num_loads)]
        for i in range(self.num_loads):
            key = f"Load{i+1}"
            d = {}
            for p in self.valid_paths.get(key, []):
                g = int(p.get("g", -1))
                if 1 <= g <= 4:
                    L = len(p.get("bits", []))
                    d[g] = min(d.get(g, 1e9), L)
            shortest_len[i] = d
        return shortest_len

    def _priority_assignment(self, action):
        n = self.num_loads
        assigned_gens = np.full(n, -1, dtype=int)
        remaining = self.G_limits.astype(float).copy()

        active_idx = [i for i in range(n) if action[i] == 1 and len(self.valid_generators[i]) > 0]

        tiers = {1: [], 2: [], 3: []}
        for i in active_idx:
            tiers[int(self.load_levels[i])].append(i)

        for L in (1, 2, 3):
            tiers[L].sort(key=lambda i: (self.loads_power[i], ))  

        dropped = []

        def try_assign_best_fit(i):
            cand = [g for g in self.valid_generators[i] if self.G_status[g-1] == 1]
            if not cand:
                return False
            need = self.loads_power[i]
            feasible = [g for g in cand if remaining[g-1] >= need - 1e-9]
            if not feasible:
                return False
            scored = []
            for g in feasible:
                residual = remaining[g-1] - need          
                path_len = self.shortest_len[i].get(g, 1e9)  
                scored.append((residual, path_len, -remaining[g-1], g))
            scored.sort()
            g_best = scored[0][-1]
            assigned_gens[i] = g_best
            remaining[g_best-1] -= need
            return True

        def try_move(i):
            g_from = assigned_gens[i]
            if g_from <= 0:
                return False
            need = self.loads_power[i]
   
            remaining[g_from-1] += need
            assigned_gens[i] = -1
            ok = try_assign_best_fit(i)
            if not ok:
                assigned_gens[i] = g_from
                remaining[g_from-1] -= need
            return ok

        for i in tiers[1]:
            if try_assign_best_fit(i):
                continue
            need = self.loads_power[i]
            cand_g = sorted([g for g in self.valid_generators[i] if self.G_status[g-1] == 1],
                            key=lambda g: remaining[g-1], reverse=True)
            placed = False
            for g in cand_g:
                if remaining[g-1] >= need - 1e-9:
                    assigned_gens[i] = g
                    remaining[g-1] -= need
                    placed = True
                    break
                deficit = need - remaining[g-1] + 1e-9
                victims = [j for j in range(n) if assigned_gens[j] == g and self.load_levels[j] >= 2]
                victims.sort(key=lambda j: (self.load_levels[j], -self.loads_power[j]))  
                freed = 0.0
                for j in victims:
                    if freed >= deficit:
                        break
                    if try_move(j):
                        freed += self.loads_power[j]
                if freed >= deficit:
                    assigned_gens[i] = g
                    remaining[g-1] -= need
                    placed = True
                    break
            if not placed:
                dropped.append(i)

        # Step-B:
        for i in tiers[2]:
            if not try_assign_best_fit(i):
                dropped.append(i)

        # Step-C
        for i in tiers[3]:
            if not try_assign_best_fit(i):
                dropped.append(i)

        return assigned_gens, dropped
