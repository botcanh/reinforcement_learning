import rclpy
from rclpy.node import Node
from time import time
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt

import sys
DATA_PATH = '/home/botcanh/dev_ws/src/reinforcement_learning/Data'
MODULES_PATH = '/home/botcanh/dev_ws/src/reinforcement_learning/reinforcement_learning'
sys.path.insert(0, MODULES_PATH)

from .Qlearning import *
from .Lidar import *
from .Control import *

# Action parameter
MIN_TIME_BETWEEN_ACTIONS = 0.0

# Initial and goal positions
INIT_POSITIONS_X = [ -0.7, -0.7, -0.5, -1, -2]
INIT_POSITIONS_Y = [ -0.7, 0.7, 1, -2, 1]
INIT_POSITIONS_THETA = [ 45, -45, -120, -90, 150]
GOAL_POSITIONS_X = [ 2.0, 2.0, 0.5, 1, 2]
GOAL_POSITIONS_Y = [ 1.0, -1.0, -1.9, 2, -1,]
GOAL_POSITIONS_THETA = [ 25.0, -40.0, -40, 60, -30,]

PATH_IND = 4

# Log file directory - Q table source
Q_TABLE_SOURCE = DATA_PATH + '/Log_learning'

XML_FILE_PATH = '/home/botcanh/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf'

class control_node(Node):
    def __init__(self):
        super().__init__('control_node')

        self.delclient = self.create_client(DeleteEntity, '/delete_entity')
        self.delresult = False

        self.spawnclient = self.create_client(SpawnEntity, '/spawn_entity')
        self.req = SpawnEntity.Request()

        self.laser_data = None
        self.odom_data = None

        self.velPub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_subscription = self.laser_subscription = self.create_subscription(LaserScan,'/scan',self.laser_listener_callback,10)
        self.odom_subcription = self.odom_subscription = self.create_subscription(Odometry,'/odom',self.odom_listener_callback,10)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.shutdown = False

        #data
        self.actions = createActions()
        self.state_space = createStateSpace()
        self.Q_table = readQTable(Q_TABLE_SOURCE+'/Qtable.csv')
        print('Init Qtable:')
        print(self.Q_table)


        #init time 
        self.t_0 = self.get_clock().now()
        self.t_start = self.get_clock().now()

        #init timer
        while not(self.t_start > self.t_0):
            self.t_start = self.get_clock().now()

        self.t_step = self.t_start
        self.count = 0

    def call_spawn_entity_service(self, name, xml_file_path, x, y, z):
        try:
            with open(xml_file_path, 'r') as file:
                xml_content = file.read()

            self.req.name = name
            self.req.xml = xml_content
            self.req.initial_pose.position.x = x
            self.req.initial_pose.position.y = y
            self.req.initial_pose.position.z = z

            future = self.spawnclient.call_async(self.req)
            future.add_done_callback(self.spawn_entity_callback)
        except Exception as e:
            self.get_logger().error('Failed to call spawn_entity service: %r' % e)

    def spawn_entity_callback(self, future):
        try:
            response = future.result()
            if response is not None:
                self.get_logger().info(
                    'Spawn entity response: success={}, status_message={}'.format(
                        response.success, response.status_message
                    )
                )
            else:
                self.get_logger().error('Service call failed')
        except Exception as e:
            self.get_logger().error('Service call failed: %r' % e)


    def delete_entity(self, name):
        request = DeleteEntity.Request()
        request.name = name

        # Call the service
        future = self.delclient.call_async(request)
        future.add_done_callback(self.delete_entity_callback)

    def delete_entity_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.delresult = True
            else:
                self.delresult = False
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    
        
    def laser_listener_callback(self, msg):
        # This function will be called when a new message is received
        self.laser_data = msg

    def odom_listener_callback(self, msg):
        self.odom_data = msg

    def timer_callback(self):
        msgScan = self.laser_data
        msgOdom = self.odom_data

        step_time = (self.get_clock().now() - self.t_step).nanoseconds * 1e-9
        if msgOdom is None:
            self.get_logger().info('Waiting for odom data...')
            return

        if step_time > MIN_TIME_BETWEEN_ACTIONS:
            step_time = self.get_clock().now()

            (x, y) = getPosition(msgOdom)
            theta = degrees(getRotation(msgOdom))
            (lidar, angles) = lidarScan(msgScan)
            (state_ind, x1, x2, x3, x4) = scanDiscretization(self.state_space , lidar)

            #check for obstacles
            crash = checkCrash(lidar)
            object_nearby = checkObjectNearby(lidar)
            goal_near = checkGoalNear(x, y, GOAL_POSITIONS_X[3], GOAL_POSITIONS_Y[3])
            if crash:
                robotStop(self.velPub)
                print('crash, end simulation')
                self.shutdown = True
            else:
                (action, status) = getBestAction(self.Q_table, state_ind, self.actions)
                if not status == 'getBestAction => OK':
                    print('\r\n', status, '\r\n')

                status = robotDoAction(self.velPub, action)
                if not status == 'robotDoAction => OK':
                    print('\r\n', status, '\r\n')
            
            if goal_near == True:
                robotStop(self.velPub)
                print("reach")

def main(args=None):
    rclpy.init(args=args)

    control = control_node()
    sleep(1)
    rclpy.spin(control)

    if control_node.shurdown == True:
        control_node.destroy_node()

    rclpy.shutdown()
