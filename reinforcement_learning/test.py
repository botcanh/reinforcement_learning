# Python program explaining 
# where() function 

import numpy as np 
from Qlearning import *


lidar = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ANGLE_MIN = 2
HORIZON_WIDTH = 3

print(lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1])
print(lidar[(ANGLE_MIN):(ANGLE_MIN + HORIZON_WIDTH)])

Q_table = np.zeros((3, 4))
print(Q_table)