#Import Libraries:
import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np         #array library
import math
import matplotlib as mpl   #used for image plotting

import neuralNetworkPred as nn

#Pre-Allocation

vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

sensor_number = 1

if clientID!=-1:  #check if client connection successful
    print('Connected to remote API server')
    
else:
    print('Connection not successful')
    sys.exit('Could not connect')


#retrieve motor  handles
errorCode,left_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
errorCode,right_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)


sensor_h=[] #empty list for handles
sensor_val=np.array([]) #empty array for sensor measurements

#for loop to retrieve sensor arrays and initiate sensors
for x in range(1,sensor_number+1):
        errorCode,sensor_handle=vrep.simxGetObjectHandle(clientID,'Proximity_sensor'+str(x),vrep.simx_opmode_oneshot_wait)
        sensor_h.append(sensor_handle) #keep list of handles        
        errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handle,vrep.simx_opmode_streaming)                
        sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values

# while (time.time()-t)<600:
while True:
    #Loop Execution
    sensor_val=np.array([])    
    for x in range(1,sensor_number+1):
        errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_h[x-1],vrep.simx_opmode_buffer)                
        sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values

    vl = 1
    vr = 1
    sleepTime = 0.2

    state = list(map(lambda x : 1 if x > 10e-5 else 0, sensor_val))
    state = np.array(state).reshape(1,sensor_number)

    choice = nn.nextChoice(state)

    if choice == 'left':
        vr = 0.5
        vl = -1
        sleepTime = 0.4
    elif choice == 'right':
        vr = -1
        vl = 0.5
        sleepTime = 0.4

    errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
    errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)


    time.sleep(sleepTime) #loop executes once every 0.2 seconds (= 5 Hz)

#Post ALlocation
errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)
    

