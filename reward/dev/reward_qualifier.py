import math

def reward_function(params):

  # Read input parameters
  track_width = params['track_width']
  distance_from_center = params['distance_from_center']
  
  # reward function as Gauss curve with the variable distance_from_center
  reward = (1 / (math.sqrt(2 * math.pi * (track_width*2/15) ** 2)) * math.exp(-((distance_from_center + track_width/10) ** 2 / (4 * track_width*2/15) ** 2))) *(track_width*2/3)
  
  return float(reward)
