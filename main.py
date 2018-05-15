import os
import time
import nxt
import neuralNetworkPred as nn
from nxt.sensor import *
from nxt.motor import *

print('looking for NXT ... could take 15 seconds')
b = nxt.locator.find_one_brick()

m_left = Motor(b, PORT_B)

m_right = Motor(b, PORT_A)

both = nxt.SynchronizedMotors(m_left, m_right, 0)
rightboth = nxt.SynchronizedMotors(m_left, m_right, 100)
leftboth = nxt.SynchronizedMotors(m_right, m_left, 100)

filecnt=0
lastAction = ''
while True:
    distance = Ultrasonic(b, PORT_4,check_compatible=False).get_sample()
    state = [0]
    if distance < 50 :
        state[0] = 1

    action = nn.nextChoice(state)

    if action == 'forward' and lastAction != 'forward':
        both.run(100)
    elif action == 'left':
        both.brake()
        leftboth.turn(100, 60, False)
    elif action == 'right':
        both.brake()
        rightboth.turn(100, 60, False)
    
    lastAction = action
    time.sleep(0.3)
    