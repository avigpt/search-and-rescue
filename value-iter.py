"""Implementation of Value Iteration"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time

NUM_EPOCHS = 100
DISC = 0.9
NUM_ACTIONS = 4 # (1 is up, 2 is right, 3 is down, 4 is left)
MAX_DENSITY = 3

num_states = None # number of states
actions_from_state = {} # {state -> possible actions in data}
sp_from_sa = {} # {state -> {action -> next state}}
rewards = {} # {state -> {action -> reward}}
density = {} # {state -> object density}
policy = {} # {state -> best action}

def parse_data(data):
    """
    Populate global variables to track model information, specifically:
    number of states, possible actions from each state, the states 
    reachable from a given state, object density for each state, and the reward of 
    a state-action pair. 
    """
    for _, row in data.iterrows():
        s = row['s']
        a = row['a']
        r = row['r']
        sp = row['sp']
        d = row['d']

        if s not in actions_from_state:
            actions_from_state[s] = set()
        actions_from_state[s].add(a)

        if s not in sp_from_sa:
            sp_from_sa[s] = {}
        sp_from_sa[s][a] = sp

        if s not in rewards:
            rewards[s] = {}
        rewards[s][a] = r

        density[s] = d

    if len(actions_from_state) != len(sp_from_sa) or len(actions_from_state) != len(rewards):
        raise Exception("Error parsing data: number of states mismatched.")
    global num_states
    num_states = len(actions_from_state)

def T(sp):
    """
    Returns transition probability of going into state 'sp' based on density.
    Caller must ensure it is possible to move into 'sp' from some state and action.
    """
    return 0.5 if density[sp] >= MAX_DENSITY - 1 else 1

def update(s, U_s):
    """
    Update U(s) and store the action that gives the best utility at this state s.
    """
    max_u = float('-inf')
    a_max = None
    for a in actions_from_state[s]:
        sp = sp_from_sa[s][a] # can only transition to one state
        tr = T(sp)
        u_sp = U_s[sp - 1]
        future = tr * u_sp
        u = rewards[s][a] + (DISC * future)
        if u > max_u:
            max_u = u
            a_max = a
    policy[s] = a_max
    return max_u

def value_iteration():
    """
    Iterate on best value obtainable at each state for NUM_EPOCHS iterations. 
    """
    U_s = [0.0] * num_states
    for _ in range(NUM_EPOCHS):
        max_diff = 0
        for s in range(1, num_states + 1):
            new_u = update(s, U_s)
            max_diff = max(max_diff, abs(new_u - U_s[s - 1]))
            U_s[s - 1] = new_u
        if max_diff <= 0.005: 
            break # convergence

def write_policy(filename):
    plot_actions = []
    with open(f"policies/{filename}", 'w') as f:
        for s in policy:
            plot_actions.append(policy[s])
            f.write(f"{policy[s]}\n")
    return plot_actions

def get_policy_grid(policy):
    actions_as_grid = []
    action_row = []
    row_len = math.sqrt(len(policy))
    for i in range(len(policy) + 1):
        if (not(i % row_len) and i != 0):
            actions_as_grid.append(action_row)
            action_row = []
        if (i != len(policy)):
            action_row.append(policy[i])
    return actions_as_grid

def plot_policy(policy_as_grid):
    arrows = {2: (1,0), 4: (-1, 0), 1: (0, 1), 3: (0, -1)}
    scale = 0.25
    fig, ax = plt.subplots(figsize=(6, 6))
    for r, row in enumerate(policy_as_grid):
        for c, cell in enumerate(row):
            plt.arrow(c, 5-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width = 0.1)
    plt.title('Value Iteration Generated Policy')
    plt.show()

def main():
    if len(sys.argv) != 3:
        raise Exception("Usage: python3 value-iter.py <infile>.csv <outfile>.policy")
    start = time.time()
    infile = sys.argv[1]
    data = pd.read_csv("data/" + infile)
    parse_data(data)
    value_iteration()
    policy = write_policy(sys.argv[2])
    end = time.time()
    print("Time: ", end - start)
    policy_as_grid = get_policy_grid(policy)
    plot_policy(policy_as_grid)


if __name__ == "__main__":
    main()