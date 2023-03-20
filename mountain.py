"""Generating Random Mountains"""
import sys
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FOUND_REWARD = 50

def generate_random_mountain(n):
    """
    Create random mountain of size n with cells that contain tuples of (height, obstacle_density). 
    Mountain was created by concatenating multiple 3x3 mountains with different heights.
    Max Height is 20 and max obstacle density is 3
    Returns a grid where each cell has (height, obstacle_density)
    """

    iterations = n // 3
    mod = n % 3
    if mod != 0: 
        extra_iter = iterations + 1
    else:
        extra_iter = iterations
    main_grid = []
    
    
    for iter in range(extra_iter):
        current_grid_row = create_grid_row(iterations, mod)
        if (mod != 0 and iter == (iterations - 1)):
            current_grid_row = current_grid_row[3 - mod:]

        cur_grid_len = len(main_grid) 
        if cur_grid_len == 0:
            main_grid = current_grid_row
        else:
            for k in range(len(current_grid_row)):
                main_grid.append(current_grid_row[k])

    
    return main_grid


def create_grid_row(iterations, mod):
    current_row_grid = []
    for _ in range(iterations):
        grid = get_3x3_mountain()
        if len(current_row_grid) == 0:
            current_row_grid = grid
        else:    
            current_row_grid = [current_row_grid[i] + grid[i] for i in range(len(grid))]
    
    remainder_grid = get_3x3_mountain()
    remainder = cut_grid(remainder_grid, mod)
    current_row_grid = [r1 + r2 for r1, r2 in zip(current_row_grid, remainder)] 

    return current_row_grid

def cut_grid(grid, n):
    """ 
    Converts a AxA grid into Axn
    """
    return [row[:n] for row in grid]

    

def get_3x3_mountain():
    height_range = 20
    obstacle_density = 3
    peak = random.randint(2, height_range)
    grid = [[(peak - 2, random.randint(0, obstacle_density)), (peak - 1, random.randint(0, obstacle_density)), (peak - 2, random.randint(0, obstacle_density))],
            [(peak - 1, random.randint(0, obstacle_density)), (peak, random.randint(0, obstacle_density)), (peak - 1, random.randint(0, obstacle_density))],
            [(peak - 2, random.randint(0, obstacle_density)), (peak - 1, random.randint(0, obstacle_density)), (peak - 2, random.randint(0, obstacle_density))]]
    
    return grid

def generate_stranded_person_location(rows, columns):
    return random.randint(1, rows * columns)

def get_actions(rows, columns, row, column):
    actions = [1,2,3,4]
    if row == 0:
        actions.remove(1)
    if row == rows - 1:
        actions.remove(3)
    if column == 0:
        actions.remove(4)
    if column == columns - 1:
        actions.remove(2)
    return actions

def get_reward(action, sp_cell, stranded_person=False):
    """
    Reward formula: height + density - fuel_cost + found
    """
    if action == 1:
        action_r = -3
    if action == 2 or action == 4:
        action_r = -2
    if action == 3:
        action_r = -1
    sp_height = sp_cell[0]
    sp_density = sp_cell[1]
    found = FOUND_REWARD if stranded_person else 0
    return sp_height + sp_density + action_r + found  

def get_sp(cell_number, rows, action):
    if action == 1:
        cell_number -= rows
    if action == 2:
        cell_number += 1
    if action == 3:
        cell_number += rows
    if action == 4:
        cell_number -= 1
    return cell_number

def get_sp_cell(sp_cell_number, rows, grid):
    sp_cell_number -= 1
    sp_i = sp_cell_number // rows
    sp_j = sp_cell_number % rows
    return grid[sp_i][sp_j]
    

def generate_mountain_data(grid):
    """
    Generate mountan_data grid where each column has the values of s, a, r, sp, d
    """
    rows = len(grid)
    columns = len(grid[0])
    cell_number = 0
    mountain_data = []
    stranded_location = generate_stranded_person_location(rows, columns)
    print('Stranded location in mountain is in cell number', stranded_location, ' (starts with 1)')
     
    for i in range(rows):
        for j in range(columns):
            cell_number += 1
            actions = get_actions(rows, columns, i, j)
            for action in actions:
                sp_cell_number = get_sp(cell_number, rows, action)
                sp_cell = get_sp_cell(sp_cell_number, rows, grid)
                r = get_reward(action, sp_cell, sp_cell_number == stranded_location)
                cell_density = grid[i][j][1]
                current_row = [cell_number, action, r, sp_cell_number, cell_density] 
                mountain_data.append(current_row)
            
    return mountain_data, stranded_location

def generate_mountain_csv(mountain_data, filename):
    filename = './data/' + str(filename) + '_mountain_data.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['s', 'a', 'r', 'sp', 'd'])
        writer.writerows(mountain_data)
    
def get_heights_densities(grid):
    all_heights = []
    all_densities = []
    prev_max_height = 0
    prev_max_density = 0
    for sector in grid:
        list_heights = [height for height, density in sector]
        list_densities = [density for height, density in sector]
        curr_max_height = max(list_heights)
        curr_max_density = max(list_densities)
        if curr_max_height > prev_max_height:
            prev_max_height = curr_max_height
        if curr_max_density > prev_max_density:
            prev_max_density = curr_max_density
        all_heights.append(list_heights)
        all_densities.append(list_densities)
    return all_heights, prev_max_height, all_densities, prev_max_density
    
def plot_mountain_height(grid, peak, stranded_location):
    data_set = np.asarray(grid)
    colormap = sns.color_palette("mako", peak)
    ax = sns.heatmap(data_set, linewidths = 0.5, cmap = colormap, annot = True)
    plt.title('Mountain Terrain Height Heat Map')
    plt.show()

def plot_mountain_density(grid, peak, stranded_location):
    data_set = np.asarray(grid)
    colormap = sns.color_palette("mako", peak)
    ax = sns.heatmap(data_set, linewidths = 0.5, cmap = colormap, annot = True)

    plt.title('Mountain Terrain Density Heat Map')
    plt.show()

def main():
    if len(sys.argv) != 2:
        raise Exception("Usage: python3 mountain.py <mountain_size>")
    
    mountain_size = int(sys.argv[1])
    print('Mountain of size', mountain_size, '...')
    grid = generate_random_mountain(mountain_size)
    all_heights = get_heights_densities(grid)

    print('Generating mountain data')
    mountain_data = generate_mountain_data(grid)
    generate_mountain_csv(mountain_data[0], mountain_size)

    print('Mountain data generated')
    plot_mountain_height(all_heights[0], all_heights[1], mountain_data[1])
    plot_mountain_density(all_heights[2], all_heights[3], mountain_data[1])

if __name__ == "__main__":
    main()
