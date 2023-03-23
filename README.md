# search-and-rescue
Final Project for CS 238: Decision Making Under Uncertainty
Contributors: Avi Gupta, Sam Kwok, Ernesto Nam

In this project, we model the problem of search and rescue (SAR) in wintery conditions on mountainous terrain. If a skier or snowboarder gets stuck or lost on a mountain, an unmanned aerial vehicle (UAV) can be deployed to maximize the efficiency of SAR operations. Rather than being operated by humans, the efficiency of SAR can be improved by autonomous decision making.

The states in our MDP models are cells in a 2-D grid that represents a 3-D mountain. Each cell state contains a height and obstacle density. With this, we can produce reward and transitions. We use Q-Learning and Value Iteration methods to solve the MDP and obtain a policy.

The reward model is: R(sp | s, a) = height(sp) + density(sp) - fuel_cost(s, sp) + found(sp). The boolean "found" represents whether the stranded skier was at cell sp (s-prime). The fuel cost is -3
for ascending, -2 for lateral movement, and -1 for descending.

The transition model is: 0.5 if density(sp) >= 2 else 1. In other words, the UAV transitions
with probability 0.5 if the tree density is high enough, representing the possibility of a crash.

Usage:
Generate a mountain model with `python3 mountain.py <mountain_size>`.
    Example output: data/3_mountain_data.csv
Run q-learning with: `python3 q.py <infile>.csv <outfile>.policy`
    Example output: policies/3q.policy
Run value iteration with: `python3 value-iter.py <infile>.csv <outfile>.policy`
    Example output: policies/3v.policy
