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


def move_gripper_to_cube_home_position(bot, left=False, right=False):
    roll_angle = 0
    home_x = 0.14
    home_pitch_adjust = 0
    # home pitch adjust is only because our px150 base motor is not functioning perfectly

    if left:
        roll_angle = -np.pi/2
        home_pitch_adjust = -0.05
        home_x = 0.125
    elif right:
        roll_angle = np.pi/2
        home_pitch_adjust = -0.05
        home_x = 0.125

    # above cube at home
    bot.arm.set_ee_pose_components(
        x=home_x,z=0.15, 
        pitch=np.pi/2  + home_pitch_adjust, 
        roll=0)        
    
    bot.arm.set_ee_pose_components(x=home_x,z=0.15, pitch=np.pi/2 + home_pitch_adjust, roll=roll_angle)

    # grab cube at home (a little bit towards the far end)
    bot.arm.set_ee_pose_components(x=home_x,z=0.045, pitch=np.pi/2+ home_pitch_adjust, roll=roll_angle)


def raise_cube_from_home_position(bot, left=False, right=False):
    roll_angle = 0
    home_x = 0.14
    home_pitch_adjust = 0
    # home pitch adjust is only because our px150 base motor is not functioning perfectly

    if left:
        roll_angle = -np.pi/2
        home_pitch_adjust = -0.05
        home_x = 0.125
    elif right:
        roll_angle = np.pi/2
        home_pitch_adjust = -0.05
        home_x = 0.125


    # raise cube directly up
    bot.arm.set_ee_pose_components(x=home_x,z=0.15, pitch=np.pi/2+ home_pitch_adjust, roll=roll_angle)

    # move cube forward and pitch it up
    bot.arm.set_ee_pose_components(x=0.2,z=0.25, pitch=0, roll=0)

def lower_cube_for_regripping(bot, bottom=False):
    roll = 0
    vertical_offset = 0
    if bottom == True:
        roll = np.pi
        vertical_offset = 0.005

    # rotate arm
    bot.arm.set_ee_pose_components(x=0.2,z=0.25, pitch=0, roll=roll)

    # lower cube and drop it
    bot.arm.set_ee_pose_components(x=0.25, z=0.065 + vertical_offset, pitch=0, roll=roll)
    bot.gripper.open()

    # move gripper back
    bot.arm.set_ee_pose_components(x=0.25,z=0.15, pitch=0, roll=roll)

    # reset the roll
    bot.arm.set_ee_pose_components(x=0.25,z=0.15, pitch=0, roll=0)

    # point gripper down
    bot.arm.set_ee_pose_components(x=0.175,z=0.07, pitch=np.pi/2, roll=0)

def grip_cube_at_x(bot, x):
    pitch_offset = -0.01
    # move onto cube 
    # add a pitch offset because of arm inconsistency
    bot.arm.set_ee_pose_components(x=x,z=0.07, pitch=np.pi/2 + pitch_offset, roll=0)
    bot.gripper.close()


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
        # grabs cube at home
        move_gripper_to_cube_home_position(bot, left=True)

        bot.gripper.close()
        
        # raises cube directly up and straightens it out
        raise_cube_from_home_position(bot, left=True)

        # place the cube down for re-gripping
        lower_cube_for_regripping(bot, bottom=True)

        # grab the cube at the current x
        current_cube_x = 0.28
        grip_cube_at_x(bot, current_cube_x)

        move_gripper_to_cube_home_position(bot)
        bot.gripper.open()

        move_gripper_to_cube_home_position(bot)


if __name__=='__main__':
    main()
