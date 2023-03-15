"""Implementation of Gauss-Seidel Value Iteration"""
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
new_states_from_state = {} # {state -> possible next states}
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
        r = row['s']
        sp = row['sp']
        d = row['d']

        if s not in actions_from_state:
            actions_from_state[s] = set()
        actions_from_state[s].add(a)

        if s not in new_states_from_state:
            new_states_from_state[s] = set()
        new_states_from_state[s].add(sp)

        if s not in rewards:
            rewards[s] = {}
        rewards[s][a] = r

        density[s] = d

    if len(actions_from_state) != len(new_states_from_state) or len(actions_from_state) != len(rewards):
        raise Exception("Error parsing data: number of states mismatched.")
    global num_states
    num_states = len(actions_from_state)

def T(sp):
    return 0.5 if density[sp] >= MAX_DENSITY - 1 else 1

def update(s, U_s):
    """
    Update U(s) and store the action that gives the best utility at this state s.
    """
    max_u = float('-inf')
    a_max = None
    # print(U_s)
    for a in actions_from_state[s]:
        future = 0
        for sp in new_states_from_state[s]:
            tr = T(sp)
            # print(sp)
            u_sp = U_s[sp - 1]
            future += tr * u_sp
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
        for s in range(1, num_states + 1):
            U_s[s - 1] = update(s, U_s)

def write_policy(filename):
    with open(f"policies/{filename}", 'w') as f:
        for s in policy:
            f.write(f"{policy[s]}\n")
def main():
    if len(sys.argv) != 3:
        raise Exception("Usage: python3 value-iter.py <infile>.csv <outfile>.policy")
    start = time.time()
    infile = sys.argv[1]
    data = pd.read_csv("data/" + infile)
    parse_data(data)
    value_iteration()
    write_policy(sys.argv[2])
    end = time.time()
    print("Time: ", end - start)

if __name__ == "__main__":
    main()