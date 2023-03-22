# search-and-rescue
Final Project for CS 238: Decision Making Under Uncertainty 

In this project, we model the problem of search and rescue (S&R) in wintery conditions on mountainous terrain. If a skier or snowboarder gets stuck or lost on a mountain, an unmanned aerial vehicle (UAV) can be deployed to maximize the efficiency of S&R operations. Rather than being operated by humans, the efficiency of S&R can be improved by autonomous decision making.

The states in our MDP models are cells in a 2-D grid that represents a flattened mountain. Each cell contains a height and obstacle density. With this, we can generate our reward and transition models. We use Q-Learning and Value Iteration methods to solve the MDP and obtain a policy.
