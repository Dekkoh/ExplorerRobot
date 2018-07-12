# import random
# import numpy as np
# from scipy.sparse import rand

# length = 100
# width = 100
# map_density = 0.2
# area_map = np.ones([length, width])

# def generateMap():
#     new_map = rand(length, width, density=map_density)
#     new_map.data[:] = 1
#     new_map = np.array(new_map.A)
#     new_map = new_map.astype(int)
#     new_map[:][0] = 1
#     new_map[:][width - 1] = 1
#     new_map[:,0] = 1
#     new_map[:,length - 1] = 1

#     global area_map
#     area_map = new_map


# def getState(current_pos, direction):

#     state = np.zeros([1,1])

#     if (direction == 1):
#         state[0][0] = area_map[current_pos[0] - 1][current_pos[1]]
#         # state[0][1] = area_map[current_pos[0]][current_pos[1] - 1]
#         # state[0][2] = area_map[current_pos[0]][current_pos[1] + 1]
#     elif (direction == 2):
#         state[0][0] = area_map[current_pos[0]][current_pos[1] + 1]
#         # state[0][1] = area_map[current_pos[0] - 1][current_pos[1]]
#         # state[0][2] = area_map[current_pos[0] + 1][current_pos[1]]
#     elif (direction == 3):
#         state[0][0] = area_map[current_pos[0] + 1][current_pos[1]]
#         # state[0][1] = area_map[current_pos[0]][current_pos[1] + 1]
#         # state[0][2] = area_map[current_pos[0]][current_pos[1] - 1]
#     elif (direction == 4):
#         state[0][0] = area_map[current_pos[0]][current_pos[1] - 1]
#         # state[0][1] = area_map[current_pos[0] + 1][current_pos[1]]
#         # state[0][2] = area_map[current_pos[0] - 1][current_pos[1]]

#     return state

# def randomInitalPos():

#     done = False
#     x_pos = -1
#     y_pos = -1

#     while(not done):
#         i = random.randint(1,len(area_map) - 2)
#         j = random.randint(1,len(area_map[0]) - 2)
#         if (area_map[i][j] == 0):
#             x_pos = i
#             y_pos = y
#             done = True

#     return [x_pos, y_pos]


# def randomInitialDir():
#     direction = random.randint(1,4)
#     return direction


# def printValidation(current_pos, direction):

#     dir_string = ['Up', 'Right', 'Down', 'Left']
#     x_pos = current_pos[0]
#     y_pos = current_pos[1]

#     print('{}\n'.format(dir_string[direction-1]))

#     s = ''

#     for i in range(len(area_map)):
#         for j in range(len(area_map[0])):
#             if (x_pos == i and y_pos == j):
#                 s = s + '*'
#             elif (area_map[i][j] == 0):
#                 s = s + '.'
#             else:
#                 s = s + str(area_map[i][j])
#             s = s + ' '
#         s = s + '\n'

#     print(s)

# def nextStep(next_choice, current_pos, direction):
#     last_pos = current_pos
#     last_dir = direction
#     last_state = getState(last_pos, last_dir)

#     if (next_choice == 'forward'):
#         if (direction == 1):
#             current_pos[0] -= 1
#         elif (direction == 2):
#             current_pos[1] += 1
#         elif (direction == 3):
#             current_pos[0] += 1
#         elif (direction == 4):
#             current_pos[1] -= 1
#     elif (next_choice == 'right'):
#         direction += 1
#         if (direction > 4):
#             direction = 1
#     elif (next_choice == 'left'):
#         direction -= 1
#         if (direction < 1):
#             direction = 4

#     state = np.zeros(1)

#     if (area_map[current_pos[0]][current_pos[1]] == 1):
#         done = True
#         direction = last_dir
#         current_pos = last_pos
#         state = last_state
#         reward = -30
#     else:
#         done = False
#         state = getState(current_pos, direction)
#         if (next_choice == 'forward'):
#             reward = 5
#         else:
#             reward = 1

#     return state, reward, done, current_pos, direction

import random
import numpy as np
from scipy.sparse import rand

length = 10
width = 10
map_density = 0.2

def generate_map():
    new_map = rand(length, width, density=map_density)
    new_map.data[:] = 1
    new_map = np.array(new_map.A)
    new_map = new_map.astype(int)
    new_map[:][0] = 1
    new_map[:][width - 1] = 1
    new_map[:,0] = 1
    new_map[:,length - 1] = 1

    return new_map


def get_state(area_map, x_pos, y_pos, direction):

    state = np.zeros([1,1])

    if (direction == 1):
        state[0][0] = area_map[x_pos - 1][y_pos]
        # state[0][1] = area_map[x_pos][y_pos - 1]
        # state[0][2] = area_map[x_pos][y_pos + 1]
    elif (direction == 2):
        state[0][0] = area_map[x_pos][y_pos + 1]
        # state[0][1] = area_map[x_pos - 1][y_pos]
        # state[0][2] = area_map[x_pos + 1][y_pos]
    elif (direction == 3):
        state[0][0] = area_map[x_pos + 1][y_pos]
        # state[0][1] = area_map[x_pos][y_pos + 1]
        # state[0][2] = area_map[x_pos][y_pos - 1]
    elif (direction == 4):
        state[0][0] = area_map[x_pos][y_pos - 1]
        # state[0][1] = area_map[x_pos + 1][y_pos]
        # state[0][2] = area_map[x_pos - 1][y_pos]

    return state

def random_inital_pos(area_map):

    done = False
    x_pos = -1
    y_pos = -1

    while(not done):
        i = random.randint(1,len(area_map) - 2)
        j = random.randint(1,len(area_map[0]) - 2)
        if (area_map[i][j] == 0):
            x_pos = i
            y_pos = j
            done = True

    return x_pos, y_pos


def random_initial_dir():
    direction = random.randint(1,4)
    return direction


def print_validation(area_map, x_pos, y_pos, direction):

    dir_string = ['Up', 'Right', 'Down', 'Left']

    print('{}\n'.format(dir_string[direction-1]))

    s = ''

    for i in range(len(area_map)):
        for j in range(len(area_map[0])):
            if (x_pos == i and y_pos == j):
                s = s + '*'
            elif (area_map[i][j] == 0):
                s = s + '.'
            else:
                s = s + str(area_map[i][j])
            s = s + ' '
        s = s + '\n'

    print(s)

def next_step(area_map, next_choice, x_pos, y_pos, direction):
    last_x_pos = x_pos
    last_y_pos = y_pos
    last_dir = direction
    last_state = get_state(area_map, x_pos, y_pos, last_dir)

    if (next_choice == 'forward'):
        if (direction == 1):
            x_pos -= 1
        elif (direction == 2):
            y_pos += 1
        elif (direction == 3):
            x_pos += 1
        elif (direction == 4):
            y_pos -= 1
    elif (next_choice == 'right'):
        direction += 1
        if (direction > 4):
            direction = 1
    elif (next_choice == 'left'):
        direction -= 1
        if (direction < 1):
            direction = 4

    state = np.zeros(1)

    if (area_map[x_pos][y_pos] == 1):
        done = True
        direction = last_dir
        x_pos = last_x_pos
        y_pos = last_y_pos
        state = last_state
        reward = -30
    else:
        done = False
        state = get_state(area_map, x_pos, y_pos, direction)
        if (next_choice == 'forward'):
            reward = 5
        else:
            reward = 1

    return state, reward, done, x_pos, y_pos, direction