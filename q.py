"""Implementation of Q-Learning"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time

NUM_ACTIONS = 4
DISC = 0.9
NUM_EPOCHS = 100
LEARN_RATE = 0.1
DIFF = 0.001

actions_from_state = {} # state -> possible actions in data

def update(row, q_table):
    s = row['s']
    a = row['a']
    r = row['r']
    sp = row['sp']
    q = q_table[s - 1][a - 1] # zero-indexed
    
    max_future = float('-inf')
    if sp not in actions_from_state: return
    for ap in actions_from_state[sp]:
        max_future = max(max_future, q_table[sp - 1][ap - 1])

    update = q + (LEARN_RATE * (r + (DISC * max_future) - q))
    q_table[s - 1][a - 1] = update
    return int(abs(q - update) < DIFF)

def populate_actions(data):
    for _, row in data.iterrows():
        s = row['s']
        a = row['a']
        if s not in actions_from_state:
            actions_from_state[s] = set()
        actions_from_state[s].add(a)

def learn(data, q_table, num_states):
    epoch = 0
    while epoch < NUM_EPOCHS:
        converge_count = 0
        for _, row in data.iterrows():
            converge_count += update(row, q_table)
            if converge_count == num_states:
                return
        epoch += 1

def write_policy(filename, q_table):
    plot_actions = []
    max_actions_i = np.argmax(q_table, axis=1)
    with open(f"policies/{filename}", 'w') as f:
        for action in max_actions_i:
            plot_actions.append(action)
            f.write(f"{action + 1}\n")
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
            plt.arrow(c, 5-r, scale*arrows[cell + 1][0], scale*arrows[cell + 1][1], head_width = 0.1)
    plt.title('Q-Learning Generated Policy')
    plt.show()

def main():
    if len(sys.argv) != 3:
        raise Exception("Usage: python3 q.py <infile>.csv <outfile>.policy")
    start = time.time()
    infile = sys.argv[1]
    data = pd.read_csv("data/" + infile)
    populate_actions(data)
    num_states = len(actions_from_state)
    q_table = np.random.rand(num_states, NUM_ACTIONS)
    learn(data, q_table, num_states)
    policy = write_policy(sys.argv[2], q_table)
    end = time.time()
    print("Time: ", end - start)
    policy_as_grid = get_policy_grid(policy)
    plot_policy(policy_as_grid)

if __name__ == "__main__":
    main()