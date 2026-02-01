# train_many.py
import math
import numpy as np
import torch
from collections import Counter

from supply_env_a2cV1 import SupplyEnv
from a2c_agent import A2CAgent
import time

G_limits = [1200, 600, 600, 1200]
switches = [1] * 26

G_status = [1, 0, 1, 1]
switches[16] = 0
switches[22] = 0

loads_power  = [72, 120, 200, 150, 160, 100, 80, 325, 185, 44, 225, 205, 110, 72, 87, 100, 205, 200]
load_levels  = [1,1,1,1,1,1,2,3,2,3,2,3,2,3,2,3,2,3]

STATE_DIM  = len(loads_power)
ACTION_DIM = STATE_DIM

NUM_EPISODES = 500
ROLLOUT_LEN  = 8
N_RUNS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mean_std_se_ci(values):
    x = np.array(values, dtype=float)
    n = len(x)
    mean = float(np.mean(x)) if n else 0.0
    std  = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se   = float(std/np.sqrt(n)) if n else 0.0
    z = 1.959964
    return mean, std, se, (mean - z*se, mean + z*se)

def train_once():
    
    env = SupplyEnv(G_status, G_limits, switches, loads_power, load_levels, horizon=32)
    agent = A2CAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,
        entropy_coef=0.02, value_coef=0.5, max_grad_norm=0.5,
        device=DEVICE,
        replay_capacity=50000, critic_replay_updates=4,
        replay_batch_size=128, min_replay_size=1000,
    )

    state = env.reset()
    episode_return = 0.0
    episode_steps  = 0

    for _ in range(1, NUM_EPISODES + 1):
        steps = 0
        while steps < ROLLOUT_LEN:
            mask = env.get_action_mask()
            action, log_prob = agent.select_action(state, mask_np=mask)

            next_state, reward, done, info = env.step(action)
            v_s = agent.value(state)
            agent.store(state, action, reward, done, log_prob, v_s, next_state)

            episode_return += reward
            episode_steps  += 1
            state = next_state
            steps += 1

            if done:
                state = env.reset()
                episode_return = 0.0
                episode_steps  = 0

        agent.update()
        
    state = env.reset()
    mask  = env.get_action_mask()
    t_start = time.time()
    action = agent.greedy_action(state, mask_np=mask)
    t_end = time.time()
    duration = t_end - t_start

    next_state, reward, done, info = env.step(action)

    GP = np.array(info["power_dist"], dtype=float)
    total = GP.sum() if GP.sum() > 0 else 1.0
    share_total = GP / total
    share_limit = GP / np.array(G_limits, dtype=float)

    assigned = np.array(info["assigned_gens"], dtype=int)  # -1 或 1..4
    load_to_gen = {f"Load{i+1}": (int(assigned[i]) if assigned[i] > 0 else -1) for i in range(STATE_DIM)}
    gen_to_loads = {g: [] for g in (1,2,3,4)}
    for i in range(STATE_DIM):
        g = assigned[i]
        if g > 0:
            gen_to_loads[int(g)].append(f"Load{i+1}")

    result = {
        "fitness": float(info["fitness_single"]),
        "reward": float(info["reward_single"]),
        "pen_over": float(info["penalty_over"]),
        "pen_infeas": float(info["penalty_infeasible"]),
        "state_vec": tuple(int(v) for v in next_state.tolist()),
        "power_dist": GP.tolist(),
        "share_total": share_total.tolist(),
        "share_limit": share_limit.tolist(),
        "load_to_gen": load_to_gen,
        "gen_to_loads": gen_to_loads,
        "time": duration
    }
    return result

def main():
    all_results = []
    for r in range(N_RUNS):
        res = train_once()
        all_results.append(res)
        print(f"[Run {r+1}/{N_RUNS}] Time={res['time']:.6}s | Reward={res['reward']:.3f}  Fitness={res['fitness']:.3f}")
        print("设备(负载) → 供电电源：")
        for i in range(STATE_DIM):
            name = f"Load{i+1}"
            g = res["load_to_gen"][name]
            gstr = f"G{g}" if g > 0 else "未供电"
            print(f"  {name:>6}  ←  {gstr}")
        print("电源 → 设备(负载)列表：")
        for g in (1,2,3,4):
            loads = res["gen_to_loads"][g]
            print(f"  G{g}: {loads}")
        print("-"*60)

    rewards  = [r["reward"]  for r in all_results]
    print(rewards)
    fitnesss = [r["fitness"] for r in all_results]
    pen_over = [r["pen_over"] for r in all_results]
    pen_infe = [r["pen_infeas"] for r in all_results]
    times    = [r["time"] for r in all_results]

    mR, sR, seR, ciR = mean_std_se_ci(rewards)
    mF, sF, seF, ciF = mean_std_se_ci(fitnesss)
    mO, sO, seO, ciO = mean_std_se_ci(pen_over)
    mI, sI, seI, ciI = mean_std_se_ci(pen_infe)
    mT, sT, seT, ciT = mean_std_se_ci(times) 

    print("\n===== 统计汇总（N 次）=====")
    print(f"Computation Time: mean={mT:.6f}s, std={sT:.6f}s")
    print(f"Reward:  mean={mR:.3f}, std={sR:.3f}, se={seR:.3f}, 95%CI=({ciR[0]:.3f}, {ciR[1]:.3f})")
    print(f"Fitness: mean={mF:.3f}, std={sF:.3f}, se={seF:.3f}, 95%CI=({ciF[0]:.3f}, {ciF[1]:.3f})")
    print(f"Penalty(overload): mean={mO:.3f}, std={sO:.3f}, se={seO:.3f}, 95%CI=({ciO[0]:.3f}, {ciO[1]:.3f})")
    print(f"Penalty(infeasible): mean={mI:.3f}, std={sI:.3f}, se={seI:.3f}, 95%CI=({ciI[0]:.3f}, {ciI[1]:.3f})")

    counter = Counter(r["state_vec"] for r in all_results)
    total_runs = sum(counter.values())
    example = {}
    for r in all_results:
        k = r["state_vec"]
        if k not in example:
            example[k] = r
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    print("\n===== 最终解出现次数（按供电状态 state 向量）=====")
    for rank, (state_tuple, cnt) in enumerate(items, 1):
        ex = example[state_tuple]
        ratio = cnt / total_runs
        print(f"\n[{rank}] 次数={cnt}（{ratio:.2%}）")
        print(f"state: {list(state_tuple)}")
        GP = np.array(ex["power_dist"], dtype=float)
        share_total = np.array(ex["share_total"], dtype=float)
        share_limit = np.array(ex["share_limit"], dtype=float)
        print("功率分配（kW）：", [round(v,1) for v in GP.tolist()])
        print("占总输出占比：   ", [f"{v:.2%}" for v in share_total.tolist()])
        print("相对上限占比：   ", [f"{v:.2%}" for v in share_limit.tolist()])

if __name__ == "__main__":
    main()
