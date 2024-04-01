from interbotix_xs_modules.arm import InterbotixManipulatorXS
import numpy as np
import sys
from time import sleep

# This script makes the end-effector perform pick, pour, and place tasks
# Note that this script may not work for every arm as it was designed for the wx250
# Make sure to adjust commanded joint positions and poses as necessary
#
# To get started, open a terminal and type 'roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=wx250'
# Then change to this directory and type 'python bartender.py  # python3 bartender.py if using ROS Noetic'

def main():
    bot = InterbotixManipulatorXS("px150", "arm", "gripper")

    left_side = True
    right_side = False

    # if (bot.arm.group_info.num_joints < 5):
    #     print('This demo requires the robot to have at least 5 joints!')
    #     sys.exit()

    bot.gripper.set_pressure(1)

    bot.gripper.open()
    # start
    bot.arm.go_to_sleep_pose()

    for i in range(2):
            

        roll_angle = 0
        home_x = 0.14
        home_pitch_adjust = 0

        if i == 1:
            left_side = False
            right_side = True
        if left_side:
            roll_angle = -np.pi/2
            home_pitch_adjust = -0.05
            home_x = 0.125
        elif right_side:
            roll_angle = np.pi/2
            home_pitch_adjust = -0.05
            home_x = 0.125

        # above cube at home
        bot.arm.set_ee_pose_components(
            x=home_x,z=0.1, 
            pitch=np.pi/2  + home_pitch_adjust, 
            roll=0)
        
    
        bot.arm.set_ee_pose_components(x=home_x,z=0.1, pitch=np.pi/2 + home_pitch_adjust, roll=roll_angle)

        # grab cube at home (a little bit towards the far end)
        bot.arm.set_ee_pose_components(x=home_x,z=0.045, pitch=np.pi/2+ home_pitch_adjust, roll=roll_angle)
        sleep(1)
        bot.gripper.close()

        # raise cube directly up
        bot.arm.set_ee_pose_components(x=home_x,z=0.15, pitch=np.pi/2+ home_pitch_adjust, roll=roll_angle)


        # move cube forward and pitch it up
        bot.arm.set_ee_pose_components(x=0.2,z=0.25, pitch=0, roll=0)

        # lower cube and drop it
        bot.arm.set_ee_pose_components(x=0.25, z=0.065, pitch=0, roll=0)
        bot.gripper.open()

        # move gripper back
        bot.arm.set_ee_pose_components(x=0.25,z=0.15, pitch=0, roll=0)

        # point gripper down
        bot.arm.set_ee_pose_components(x=0.175,z=0.07, pitch=np.pi/2, roll=0)

        # move onto cube (pitched - a little bit because of arm inconsistency)
        bot.arm.set_ee_pose_components(x=0.28,z=0.07, pitch=np.pi/2 - 0.1, roll=0)
        bot.gripper.close()

        # return to home (first above)


        bot.arm.set_ee_pose_components(
            x=0.13,z=0.15, 
            pitch=np.pi/2, 
            roll=0)
        

        bot.arm.set_ee_pose_components(
            x=0.13,z=0.045, 
            pitch=np.pi/2, 
            roll=0)
        bot.arm.set_ee_pose_components(x=0.14,z=0.05, pitch=np.pi/2, roll=0)

        # sleep(3)
        bot.gripper.open()
        bot.arm.go_to_sleep_pose()




    # bot.arm.set_ee_pose_components(x=0.20,z=0.03, pitch=np.pi/2, roll=0)

    # bot.arm.set_ee_cartesian_trajectory(roll=np.pi)
    # bot.arm.set_ee_pose_components(x=0.27, z=0.07, pitch=0, roll=np.pi)

    # bot.gripper.close()

    # bot.arm.go_to_home_pose()
    return

    bot.gripper.open()




    roll_angle = 0
    front = False
    back = False
    if front or back: 
        roll_angle = np.pi 

    bot.arm.set_ee_pose_components(x=0.15, z=0.16, pitch=np.pi/2, roll=roll_angle)
    bot.arm.set_ee_pose_components(x=0.15, z=0.03, pitch=np.pi/2, roll=roll_angle)
    bot.gripper.set_pressure(0.8)
    bot.gripper.close()

    bot.arm.set_ee_pose_components(x=0.15, z=0.16, pitch=np.pi/2, roll=roll_angle)
    bot.arm.set_ee_pose_components(x=0.27, z=0.06, pitch=0, roll=roll_angle)
    bot.gripper.open()
    # bot.arm.set_ee_pose_components(x=0.27, z=0.15, pitch=0, roll=roll_angle)
    # bot.arm.set_ee_pose_components(x=0.27, z=0.05, pitch=np.pi/2, roll=roll_angle)
    # bot.gripper.close()
    # bot.arm.set_ee_pose_components(x=0.15, z=0.16, pitch=np.pi/2, roll=roll_angle)
    # bot.arm.set_ee_pose_components(x=0.27, z=0.05, pitch=0, roll=roll_angle)
    # bot.gripper.open()




    # bot.arm.set_ee_pose_components(x=0.2, z=0.1, pitch=np.pi/2)

    # bot.arm.set_ee_pose_components(x=0.25, z=0.3, pitch=0)

    
    # rotation = 1
    # angle = np.pi/2 * rotation 
    # bot.arm.set_ee_pose_components(x=0.25, z=0.3, pitch=0, roll=angle)
    # bot.arm.set_ee_cartesian_trajectory(z=-0.2)


    # bot.arm.set_ee_pose_components(x=0.2, z=0.01, pitch=0, roll=angle)

    # bot.gripper.open()
    # bot.arm.set_ee_pose_components(x=0.25, z=0.01, roll=angle)




    # bot.arm.set_ee_pose_components(x=0.3, z=0.2)
    # bot.arm.set_single_joint_position("waist", np.pi/2.0)
    # bot.gripper.open()
    # bot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    # bot.gripper.close()
    # bot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    # bot.arm.set_single_joint_position("waist", -np.pi/2.0)
    # bot.arm.set_ee_cartesian_trajectory(pitch=1.5)
    # bot.arm.set_ee_cartesian_trajectory(pitch=-1.5)
    # bot.arm.set_single_joint_position("waist", np.pi/2.0)
    # bot.arm.set_ee_cartesian_trajectory(x=0.1, z=-0.16)
    # bot.gripper.open()
    # bot.arm.set_ee_cartesian_trajectory(x=-0.1, z=0.16)
    # bot.arm.go_to_home_pose()
    # bot.arm.go_to_sleep_pose()

if __name__=='__main__':
    main()

