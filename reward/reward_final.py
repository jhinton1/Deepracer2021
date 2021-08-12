import math


def reward_function(params):

    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    is_left_of_center = params['is_left_of_center']
    steering_angle = params['steering_angle']
    speed = params['speed']
    
    if is_left_of_center == True:
        distance_from_center *= -1

    # implementation of reward function for distance from center
    reward = (1 / (math.sqrt(2 * math.pi * (track_width*2/15) ** 2)) * math.exp(-((
            distance_from_center + track_width/20) ** 2 / (4 * track_width*2/15) ** 2))) *(track_width*1/3)
    
    if not all_wheels_on_track:
        reward = 1e-3

    # implementation of reward function for steering angle
    STEERING_THRESHOLD = 14.4
    
    if abs(steering_angle) < STEERING_THRESHOLD:
        steering_reward = math.sqrt(- (8 ** 2 + steering_angle ** 2) + math.sqrt(4 * 8 ** 2 * steering_angle ** 2 + (12 ** 2) ** 2) ) / 10
    else:
        steering_reward = 0

    # aditional reward if the car is not steering too much
    reward *= steering_reward

    # reward for the car taking fast actions (speed is in m/s)
    reward *= math.sin(speed/math.pi * 5/6)
    
    # same reward for going slow with greater steering angle then going fast straight ahead 
    reward *= math.sin(0.4949 * (0.475 * (speed - 1.5241) + 0.5111 * steering_angle ** 2))

    return float(reward)
    
