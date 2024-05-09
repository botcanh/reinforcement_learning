import rclpy
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.insert(0, '/home/botcanh/dev_ws/src/reinforcement_learning/reinforcement_learning')

from Qlearning import *
from Lidar import *
from Control import *

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
HORIZON_WIDTH = 75

MIN_TIME_BETWEEN_SCANS = 0
MAX_SIMULATION_TIME = float('inf')
