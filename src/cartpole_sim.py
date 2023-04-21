import pybullet as p
import pybullet_data
import math
import time
import numpy as np
import os
from casadi import *

from cartpole_MPC import cartpole_MPC

sampling_time = 0.02 # 100hz
T = 50 # horizon

max_input = 35
x_max = 1.0

def main():
    x = np.array([0.0, math.pi, 0.0, 0.0]) # initial state
    x_ref = np.array([0.0, math.pi, 0.0, 0.0])   # target state

    # mpc setup
    mpc = cartpole_MPC(sampling_time*T, T, max_input, x_max)
    mpc.init(x0=x, xref=x_ref)


    # pybullet setup
    p.connect(p.GUI)
    #p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # load model
    cartpole = p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/cartpole.urdf", [0, 0, 0])

    textColor = [1, 1, 1]
    shift = 0.05
    p.addUserDebugText("MPC", [shift, 0, .1],
                    textColor,
                    parentObjectUniqueId=cartpole,
                    parentLinkIndex=1)

    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(False)
    p.setTimeStep(sampling_time)

    for i in range(p.getNumJoints(cartpole)):
        #disable default constraint-based motors
        p.setJointMotorControl2(cartpole, i, p.POSITION_CONTROL, targetPosition=0, force=0)
        info = p.getJointInfo(cartpole, i)
        jointName = info[1]
        jointType = info[2]
        print(jointName, jointType)

    desiredPosCartId = p.addUserDebugParameter("desiredPosCart", -1, 1, 0)

    while p.isConnected():
        desiredPosCart = p.readUserDebugParameter(desiredPosCartId)
        x_ref[0] = desiredPosCart

        # solve mpc
        current_time = time.time()
        u = mpc.solve(x, x_ref)
        print(str(math.floor(1/(time.time() - current_time))) + "hz")

        # input force
        p.setJointMotorControl2(bodyUniqueId=cartpole,
                                jointIndex=0,
                                controlMode=p.TORQUE_CONTROL,
                                force=u[0])

        # state update
        cart_info = p.getJointState(cartpole, 0)
        pole_info = p.getJointState(cartpole, 1)
        print("input=", u[0], "position=", cart_info[0], "angle=", pole_info[0])
        x = np.array([cart_info[0], pole_info[0]+math.pi, cart_info[1], pole_info[1]])

        p.stepSimulation()
        time.sleep(sampling_time)


if __name__ == '__main__':
    main()
