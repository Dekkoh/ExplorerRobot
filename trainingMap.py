import random
import numpy as np
from scipy.sparse import rand

mapSize = [100, 100]
mapDensity = 0.2
areaMap = np.ones([mapSize[0], mapSize[1]])

def generateMap():
    newMap = rand(mapSize[0], mapSize[1], density=mapDensity)
    newMap.data[:] = 1
    newMap = np.array(newMap.A)
    newMap = newMap.astype(int)
    newMap[:][0] = 1
    newMap[:][mapSize[1] - 1] = 1
    newMap[:,0] = 1
    newMap[:,mapSize[0] - 1] = 1

    global areaMap
    areaMap = newMap


def getState(currentPos, direction):

    state = np.zeros([1,1])

    if (direction == 1):
        state[0][0] = areaMap[currentPos[0] - 1][currentPos[1]]
        # state[0][1] = areaMap[currentPos[0]][currentPos[1] - 1]
        # state[0][2] = areaMap[currentPos[0]][currentPos[1] + 1]
    elif (direction == 2):
        state[0][0] = areaMap[currentPos[0]][currentPos[1] + 1]
        # state[0][1] = areaMap[currentPos[0] - 1][currentPos[1]]
        # state[0][2] = areaMap[currentPos[0] + 1][currentPos[1]]
    elif (direction == 3):
        state[0][0] = areaMap[currentPos[0] + 1][currentPos[1]]
        # state[0][1] = areaMap[currentPos[0]][currentPos[1] + 1]
        # state[0][2] = areaMap[currentPos[0]][currentPos[1] - 1]
    elif (direction == 4):
        state[0][0] = areaMap[currentPos[0]][currentPos[1] - 1]
        # state[0][1] = areaMap[currentPos[0] + 1][currentPos[1]]
        # state[0][2] = areaMap[currentPos[0] - 1][currentPos[1]]

    return state

def randomInitalPos():

    done = False
    currentPos = [-1, -1]

    while(not done):
        i = random.randint(1,len(areaMap) - 2)
        j = random.randint(1,len(areaMap[0]) - 2)
        if (areaMap[i][j] == 0):
            currentPos = [i, j]
            done = True

    return currentPos


def randomInitialDir():
    direction = random.randint(1,4)
    return direction


def printValidation(currentPos, direction):

    dirString = ['Up', 'Right', 'Down', 'Left']

    print('{}\n'.format(dirString[direction-1]))

    s = ''

    for i in range(len(areaMap)):
        for j in range(len(areaMap[0])):
            if (currentPos[0] == i and currentPos[1] == j):
                s = s + '*'
            elif (areaMap[i][j] == 0):
                s = s + '.'
            else:
                s = s + str(areaMap[i][j])
            s = s + ' '
        s = s + '\n'

    print(s)

def nextStep(nextChoice, currentPos, direction):
    lastPos = currentPos
    lastDir = direction
    lastState = getState(lastPos, lastDir)

    if (nextChoice == 'forward'):
        if (direction == 1):
            currentPos[0] -= 1
        elif (direction == 2):
            currentPos[1] += 1
        elif (direction == 3):
            currentPos[0] += 1
        elif (direction == 4):
            currentPos[1] -= 1
    elif (nextChoice == 'right'):
        direction += 1
        if (direction > 4):
            direction = 1
    elif (nextChoice == 'left'):
        direction -= 1
        if (direction < 1):
            direction = 4

    state = np.zeros(1)

    if (areaMap[currentPos[0]][currentPos[1]] == 1):
        done = True
        direction = lastDir
        currentPos = lastPos
        state = lastState
        reward = -30
    else:
        done = False
        state = getState(currentPos, direction)
        if (nextChoice == 'forward'):
            reward = 5
        else:
            reward = 1

    return state, reward, done, currentPos, direction