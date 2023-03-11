"""Implementation of Q-Learning"""
import numpy as np
import pandas as pd
import sys
import time

NUM_STATES = 9
NUM_ACTIONS = 4
DISC = 0.95
NUM_EPOCHS = 100
LEARN_RATE = 0.1
DIFF = 0.001

actions_from_state = {} # state -> possible actions in data
q_table = np.random.rand(NUM_STATES, NUM_ACTIONS)

def update(row):
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

def learn(data):
    epoch = 0
    while epoch < NUM_EPOCHS:
        converge_count = 0
        for _, row in data.iterrows():
            converge_count += update(row)
            if converge_count == NUM_STATES:
                return
        epoch += 1

def write_policy(filename):
    max_actions_i = np.argmax(q_table, axis=1)
    with open(filename, 'w') as f:
        for action in max_actions_i:
            f.write(f"{action + 1}\n")

def main():
    if len(sys.argv) != 3:
        raise Exception("Usage: python3 q.py <infile>.csv <outfile>.policy")
    start = time.time()
    infile = sys.argv[1]
    data = pd.read_csv("data/" + infile)
    populate_actions(data)
    learn(data)
    write_policy(sys.argv[2])
    end = time.time()
    print("Time: ", end - start)

if __name__ == "__main__":
    main()