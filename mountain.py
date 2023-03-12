import sys
import random
import csv

def generate_random_mountain(n):
    # n is size of mountain
    # each cell (x,y) of the mountain has x as height and y as object density
    height_range = 20
    obstacle_density = 3
    grid = [[(random.randint(0, height_range), random.randint(0, obstacle_density)) for j in range(n)] for i in range(n)]
    return grid

def generate_stranded_person_location(rows, columns):
    return random.randint(0, rows * columns)

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
    if action == 1:
        action_r = -3
    if action == 2 or action == 4:
        action_r = -2
    if action == 3:
        action_r = -1
    sp_height = sp_cell[0]
    sp_density = sp_cell[1]
    found = 10 if stranded_person else 0
    return sp_height + sp_density + action_r + found  # height + density - fuel_cost + found

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
    rows = len(grid)
    columns = len(grid[0])
    cell_number = 0
    mountain_data = []
    stranded_location = generate_stranded_person_location(rows, columns)
    for i in range(rows):
        for j in range(columns):
            cell_number += 1
            actions = get_actions(rows, columns, i, j)
            for action in actions:
                sp_cell_number = get_sp(cell_number, rows, action)
                sp_cell = get_sp_cell(sp_cell_number, rows, grid)
                r = get_reward(action, sp_cell, sp_cell == stranded_location)
                cell_density = grid[i][j][1]
                current_row = [cell_number, action, r, sp_cell_number, cell_density] # s, a, r, sp, d
                print('cell number is ', cell_number)
                print('action is ', action)
                print(current_row)
                mountain_data.append(current_row)
            
    return mountain_data

def generate_mountain_csv(mountain_data, filename):
    filename = './data/' + str(filename) + '_mountain_data.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['s', 'a', 'r', 'sp', 'density'])
        writer.writerows(mountain_data)
    

def main():
    if len(sys.argv) != 2:
        raise Exception("Usage: python3 mountain.py <mountain_size>")
    
    mountain_size = int(sys.argv[1])
    grid = generate_random_mountain(mountain_size)
    mountain_data = generate_mountain_data(grid)
    generate_mountain_csv(mountain_data, mountain_size)

if __name__ == "__main__":
    main()
