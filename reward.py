import csv
import math
import pandas as pd
import sys
import random

NUM_TRIALS = 100000

def read_policy(policy_data):
    state = 1
    actions_dict = {}
    with open(policy_data, 'r') as file:
        read_csv = csv.reader(file)
        for row in read_csv:
            actions_dict[state] = int(row[0])
            state += 1
    return actions_dict

def create_rewards_dict(states_data):
    rewards_given_action = {}
    for _, row in states_data.iterrows():
        rewards_given_action[row['s'], row['a']] = row['r'], row['sp']
    return rewards_given_action

def generate_trials(actions_dict, rewards_dict, terminal_state):
    reach_skier = len(actions_dict) // 2
    total_reward = 0
    for _ in range(NUM_TRIALS):
        state = random.randint(1, len(actions_dict))
        for _ in range(reach_skier):
            action = actions_dict[state]
            reward, next_state = rewards_dict[state, action]
            total_reward += reward
            state = next_state
    return total_reward / NUM_TRIALS
    
def generate_random(num_states):
    dimension = math.sqrt(num_states)
    random_dict = {}
    for i in range(num_states):
        num_state = i + 1
        if num_state == 1:
            random_dict[num_state] = random.choice([2, 3])
        elif num_state == dimension:
            random_dict[num_state] = random.choice([3, 4])
        elif num_state == num_states:
            random_dict[num_state] = random.choice([1, 4])
        elif num_state == num_states - dimension + 1:
            random_dict[num_state] = random.choice([1, 2])
        elif num_state <= dimension:
            random_dict[num_state] = random.choice([2, 3, 4])
        elif num_state > num_states - dimension:
            random_dict[num_state] = random.choice([1, 2, 4])
        elif not (num_state % dimension):
            random_dict[num_state] = random.choice([1, 3, 4])
        elif not ((num_state + (dimension - 1)) % dimension):
            random_dict[num_state] = random.choice([1, 2, 3])
        else:
            random_dict[num_state] = random.choice([1, 2, 3, 4])
    return random_dict

def main():
    if len(sys.argv) != 4:
        raise Exception("Usage: python3 reward.py <state_data>.csv <policy_data>.csv stranded_location")
    states = sys.argv[1]
    policy = sys.argv[2]
    terminal_state = sys.argv[3]
    states_data = pd.read_csv("data/" + states)
    actions_dict = read_policy("policies/" + policy)
    rewards_dict = create_rewards_dict(states_data)
    policy_score = generate_trials(actions_dict, rewards_dict, terminal_state)
    random_dict = generate_random(len(actions_dict))
    random_score = generate_trials(random_dict, rewards_dict, terminal_state)
    print("Policy score: " + str(policy_score))
    print("Random score: " + str(random_score))
    print("Difference: " + str(policy_score - random_score))

if __name__ == "__main__":
    main()