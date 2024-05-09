#! /usr/bin/env python3

import rclpy
from rclpy.time import Time
from rclpy.node import Node
from datetime import datetime
from std_srvs.srv import Trigger
import time
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ModelState

import sys
sys.path.append('.')

import random

import rclpy.time 
DATA_PATH = '/home/botcanh/dev_ws/src/reinforcement_learning/Data'
MODULES_PATH = '/home/botcanh/dev_ws/src/reinforcement_learning/reinforcement_learning'

from .Qlearning import *
from .Lidar import *
from .Control import *

#Episodes params
MAX_EPISODES = 1000 #default 400
MAX_STEPS_PER_EPISODE = 600 #default 500
MIN_TIME_BETWEEN_ACTIONS = 0.0

#Learning params
ALPHA = 0.5
GAMMA = 0.9


T_INIT = 25
T_GRAD = 0.95
T_MIN = 0.001

EPSILON_INIT = 0.9
EPSILON_GRAD = 0.96
EPSILON_MIN = 0.05

# 1 - Softmax , 2 - Epsilon greedy
EXPLORATION_FUNCTION = 1

# Initial position
X_INIT = -1.99
Y_INIT = -0.5
INIT_POSITIONS_X = [-1.99 ,  -0.7, -0.7, 0.5, -1.0, -2.0, -1.9, 1.5, 1.7, 0.47, -0.52]
INIT_POSITIONS_Y = [-0.5, -0.7, 0.7, 1.0 , 2.0, 1.0, -0.5, 0.7, -0.7, -1.0, 0.0]
INIT_POSITIONS_THETA = [0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
THETA_INIT = 0.0

RANDOM_INIT_POS = False

# Log file directory
LOG_FILE_DIR = DATA_PATH + '/Log_learning'

# Q table source file
Q_SOURCE_DIR = ''

XML_FILE_PATH = '/home/botcanh/turtlebot3_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf'

class LearningNode(Node):

    def __init__(self):
        super().__init__('Learning_node')

        self.delclient = self.create_client(DeleteEntity, '/delete_entity')
        self.delresult = False

        self.spawnclient = self.create_client(SpawnEntity, '/spawn_entity')
        self.req = SpawnEntity.Request()

        self.laser_data = None
        self.odom_data = None

        self.laser_subscription = self.laser_subscription = self.create_subscription(LaserScan,'/scan',self.laser_listener_callback,10)
        self.odom_subcription = self.odom_subscription = self.create_subscription(Odometry,'/odom',self.odom_listener_callback,10)
        self.setPosPub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)
        self.velPub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.callShutdown = False

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.random_ind = 0

        global actions, state_space, Q_table
        global T, EPSILON, alpha, gamma
        global ep_steps, ep_reward, episode, steps_per_episode, reward_per_episode, T_per_episode, EPSILON_per_episode
        global ep_reward_arr, reward_max_per_episode, reward_min_per_episode, reward_avg_per_episode
        global crash, t_ep, t_per_episode, t_sim_start, t_step
        global log_sim_info, log_sim_params
        global now_start, now_stop
        global robot_in_pos, first_action_taken

        self.initLearning()
        self.initParams()

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


    def initLearning(self):
        global actions, state_space, Q_table
        actions = createActions()
        state_space = createStateSpace()
        if Q_SOURCE_DIR != '':
            Q_table = readQTable(Q_SOURCE_DIR+'/Qtable.csv')
        else:
            Q_table = createQTable(len(state_space), len(actions))
        print('Init Q_table:')
        print(Q_table)

    def initParams(self):
        global T, EPSILON, alpha, gamma
        global ep_steps, ep_reward, episode, steps_per_episode, reward_per_episode, T_per_episode, EPSILON_per_episode
        global ep_reward_arr, reward_max_per_episode, reward_min_per_episode, reward_avg_per_episode
        global crash, t_ep, t_per_episode, t_sim_start, t_step
        global log_sim_info, log_sim_params
        global now_start, now_stop
        global robot_in_pos, first_action_taken

        # Init log files
        log_sim_info = open(LOG_FILE_DIR+'/LogInfo.txt','w+')
        log_sim_params = open(LOG_FILE_DIR+'/LogParams.txt','w+')

        # Learning parameters
        T = T_INIT
        EPSILON = EPSILON_INIT
        alpha = ALPHA
        gamma = GAMMA

        #Episodes, steps, rewards
        ep_steps = 0
        ep_reward = 0
        episode = 1
        crash = 0
        reward_max_per_episode = np.array([])
        reward_min_per_episode = np.array([])
        reward_avg_per_episode = np.array([])
        ep_reward_arr = np.array([])
        steps_per_episode = np.array([])
        reward_per_episode = np.array([])

        #initial position
        robot_in_pos = False
        first_action_taken = False

        #init_time
        t_0 = self.get_clock().now()
        t_start = self.get_clock().now()

        #init timer
        while not(t_start > t_0):
            t_start = self.get_clock().now()

        t_ep = t_start
        t_sim_start = t_start
        t_step = t_start

        T_per_episode = np.array([])
        EPSILON_per_episode = np.array([])
        t_per_episode = np.array([])

        #Date 
        now_start = datetime.now()
        dt_string_start = now_start.strftime("%d/%m/%Y %H:%M:%S")

        #log date to files
        text = '\r\n' + 'SIMULATION START ==> ' + dt_string_start + '\r\n\r\n'
        print(text)
        log_sim_info.write(text)
        log_sim_params.write(text)

        # Log simulation parameters
        text = '\r\nSimulation parameters: \r\n'
        text = text + '--------------------------------------- \r\n'
        if RANDOM_INIT_POS:
            text = text + 'INITIAL POSITION = RANDOM \r\n'
        else:
            text = text + 'INITIAL POSITION = ( %.2f , %.2f , %.2f ) \r\n' % (X_INIT,Y_INIT,THETA_INIT)
        text = text + '--------------------------------------- \r\n'
        text = text + 'MAX_EPISODES = %d \r\n' % MAX_EPISODES
        text = text + 'MAX_STEPS_PER_EPISODE = %d \r\n' % MAX_STEPS_PER_EPISODE
        text = text + 'MIN_TIME_BETWEEN_ACTIONS = %.2f s \r\n' % MIN_TIME_BETWEEN_ACTIONS
        text = text + '--------------------------------------- \r\n'
        text = text + 'ALPHA = %.2f \r\n' % ALPHA
        text = text + 'GAMMA = %.2f \r\n' % GAMMA
        if EXPLORATION_FUNCTION == 1:
            text = text + 'T_INIT = %.3f \r\n' % T_INIT
            text = text + 'T_GRAD = %.3f \r\n' % T_GRAD
            text = text + 'T_MIN = %.3f \r\n' % T_MIN
        else:
            text = text + 'EPSILON_INIT = %.3f \r\n' % EPSILON_INIT
            text = text + 'EPSILON_GRAD = %.3f \r\n' % EPSILON_GRAD
            text = text + 'EPSILON_MIN = %.3f \r\n' % EPSILON_MIN

        text = text + '--------------------------------------- \r\n'
        text = text + 'MAX_LIDAR_DISTANCE = %.2f \r\n' % MAX_LIDAR_DISTANCE
        text = text + 'COLLISION_DISTANCE = %.2f \r\n' % COLLISION_DISTANCE
        text = text + 'ZONE_0_LENGTH = %.2f \r\n' % ZONE_0_LENGTH
        text = text + 'ZONE_1_LENGTH = %.2f \r\n' % ZONE_1_LENGTH
        text = text + '--------------------------------------- \r\n'
        text = text + 'CONST_LINEAR_SPEED_FORWARD = %.3f \r\n' % CONST_LINEAR_SPEED_FORWARD
        text = text + 'CONST_ANGULAR_SPEED_FORWARD = %.3f \r\n' % CONST_ANGULAR_SPEED_FORWARD
        text = text + 'CONST_LINEAR_SPEED_TURN = %.3f \r\n' % CONST_LINEAR_SPEED_TURN
        text = text + 'CONST_ANGULAR_SPEED_TURN = %.3f \r\n' % CONST_ANGULAR_SPEED_TURN
        log_sim_params.write(text)

    def timer_callback(self):
        global actions, state_space, Q_table
        global T, EPSILON, alpha, gamma
        global ep_steps, ep_reward, episode, steps_per_episode, reward_per_episode, T_per_episode, EPSILON_per_episode
        global ep_reward_arr, reward_max_per_episode, reward_min_per_episode, reward_avg_per_episode
        global crash, t_ep, t_per_episode, t_sim_start, t_step
        global log_sim_info, log_sim_params
        global now_start, now_stop
        global robot_in_pos, first_action_taken
        global prev_action, prev_lidar, prev_state_ind,lidar,action

        
        msgScan = self.laser_data
        if msgScan is None:
            self.get_logger().info('Waiting for laser data...')
            return
        step_time = (self.get_clock().now() - t_step).nanoseconds * 1e-9 # to second
        if step_time > MIN_TIME_BETWEEN_ACTIONS:
            t_step = self.get_clock().now()
            if step_time > 2:
                text = '\r\nTOO BIG STEP TIME: %.2f s' % step_time
                print(text)
                log_sim_info.write(text+'\r\n')
            
            #end of learning
            if episode > MAX_EPISODES:
                # simulation time
                sim_time = (self.get_clock().now() - t_sim_start).nanoseconds * 1e-9
                sim_time_h = sim_time // 3600
                sim_time_m = ( sim_time - sim_time_h * 3600 ) // 60
                sim_time_s = sim_time - sim_time_h * 3600 - sim_time_m * 60

                # real time
                now_stop = datetime.now()
                dt_string_stop = now_stop.strftime("%d/%m/%Y %H:%M:%S")
                real_time_delta = (now_stop - now_start).total_seconds()
                real_time_h = real_time_delta // 3600
                real_time_m = ( real_time_delta - real_time_h * 3600 ) // 60
                real_time_s = real_time_delta - real_time_h * 3600 - real_time_m * 60

                # Log learning session info to file
                text = '--------------------------------------- \r\n\r\n'
                text = text + 'MAX EPISODES REACHED(%d), LEARNING FINISHED ==> ' % MAX_EPISODES + dt_string_stop + '\r\n'
                text = text + 'Simulation time: %d:%d:%d  h/m/s \r\n' % (sim_time_h, sim_time_m, sim_time_s)
                text = text + 'Real time: %d:%d:%d  h/m/s \r\n' % (real_time_h, real_time_m, real_time_s)
                print(text)
                log_sim_info.write('\r\n'+text+'\r\n')
                log_sim_params.write(text+'\r\n')


                # Log data to file
                saveQTable(LOG_FILE_DIR+'/Qtable.csv', Q_table)
                np.savetxt(LOG_FILE_DIR+'/StateSpace.csv', state_space, '%d')
                np.savetxt(LOG_FILE_DIR+'/steps_per_episode.csv', steps_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/reward_per_episode.csv', reward_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/T_per_episode.csv', T_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/EPSILON_per_episode.csv', EPSILON_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/reward_min_per_episode.csv', reward_min_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/reward_max_per_episode.csv', reward_max_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/reward_avg_per_episode.csv', reward_avg_per_episode, delimiter = ' , ')
                np.savetxt(LOG_FILE_DIR+'/t_per_episode.csv', t_per_episode, delimiter = ' , ')

                # Close files and shut down node
                log_sim_info.close()
                log_sim_params.close()
                print("End of learning")
                self.callShutdown = True
            else:
                ep_time = (self.get_clock().now() - t_ep).nanoseconds * 1e-9
                # End of an episode
                if crash or ep_steps >= MAX_STEPS_PER_EPISODE:
                    print("crash!!! or end")
                    robotStop(self.velPub)
                    if crash:
                        # get crash position
                        odomMsg = self.odom_data
                        ( x_crash , y_crash ) = getPosition(odomMsg)
                        theta_crash = degrees(getRotation(odomMsg))

                    t_ep = self.get_clock().now()
                    reward_min = np.min(ep_reward_arr)
                    reward_max = np.max(ep_reward_arr)
                    reward_avg = np.mean(ep_reward_arr)
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

                    text = '---------------------------------------\r\n'
                    if crash:
                        text = text + '\r\nEpisode %d ==> CRASH {%.2f,%.2f,%.2f}    ' % (episode, x_crash, y_crash, theta_crash) + dt_string
                    elif ep_steps >= MAX_STEPS_PER_EPISODE:
                        text = text + '\r\nEpisode %d ==> MAX STEPS PER EPISODE REACHED {%d}    ' % (episode, MAX_STEPS_PER_EPISODE) + dt_string
                    else:
                        text = text + '\r\nEpisode %d ==> UNKNOWN TERMINAL CASE    ' % episode + dt_string
                    text = text + '\r\nepisode time: %.2f s (avg step: %.2f s) \r\n' % (ep_time, ep_time / ep_steps)
                    text = text + 'episode steps: %d \r\n' % ep_steps
                    text = text + 'episode reward: %.2f \r\n' % ep_reward
                    text = text + 'episode max | avg | min reward: %.2f | %.2f | %.2f \r\n' % (reward_max, reward_avg, reward_min)
                    if EXPLORATION_FUNCTION == 1:
                        text = text + 'T = %f \r\n' % T
                    else:
                        text = text + 'EPSILON = %f \r\n' % EPSILON
                    print(text)
                    log_sim_info.write('\r\n'+text)

                    steps_per_episode = np.append(steps_per_episode, ep_steps)
                    reward_per_episode = np.append(reward_per_episode, ep_reward)
                    T_per_episode = np.append(T_per_episode, T)
                    EPSILON_per_episode = np.append(EPSILON_per_episode, EPSILON)
                    t_per_episode = np.append(t_per_episode, ep_time)
                    reward_min_per_episode = np.append(reward_min_per_episode, reward_min)
                    reward_max_per_episode = np.append(reward_max_per_episode, reward_max)
                    reward_avg_per_episode = np.append(reward_avg_per_episode, reward_avg)
                    ep_reward_arr = np.array([])
                    ep_steps = 0
                    ep_reward = 0
                    crash = 0
                    robot_in_pos = False
                    first_action_taken = False
                    if T > T_MIN:
                        T = T_GRAD * T
                    if EPSILON > EPSILON_MIN:
                        EPSILON = EPSILON_GRAD * EPSILON
                    episode = episode + 1
                    self.delete_entity('burger')
                    self.random_ind = random.randint(0, 10)
                    self.call_spawn_entity_service('burger', XML_FILE_PATH, INIT_POSITIONS_X[self.random_ind], INIT_POSITIONS_Y[self.random_ind], INIT_POSITIONS_THETA[self.random_ind])
                    
                else:
                    ep_steps = ep_steps + 1
                    # Initial position
                    if not robot_in_pos:
                        print('robot not in position')
                        #self.call_spawn_entity_service('burger', XML_FILE_PATH, X_INIT, Y_INIT, THETA_INIT)
                        robotStop(self.velPub)
                        ep_steps = ep_steps - 1
                        first_action_taken = False
                        # init pos
                        #if RANDOM_INIT_POS:
                        #    (x_init , y_init , theta_init) = robotSetRandomPos(self.setPosPub)
                        #else:
                        #    (x_init , y_init , theta_init) = robotSetPos(self.setPosPub, X_INIT, Y_INIT, THETA_INIT)
                        #self.delete_entity('burger')
                        self.call_spawn_entity_service('burger', XML_FILE_PATH, INIT_POSITIONS_X[self.random_ind], INIT_POSITIONS_Y[self.random_ind], INIT_POSITIONS_THETA[self.random_ind])
                        (x_init , y_init , theta_init) = INIT_POSITIONS_X[self.random_ind], INIT_POSITIONS_Y[self.random_ind], INIT_POSITIONS_THETA[self.random_ind]
                        odomMsg = self.odom_data 
                        if odomMsg is None:
                            print("cannot get data")
                        (x, y) = getPosition(odomMsg)
                        print((x, y))
                        theta = degrees(getRotation(odomMsg))
                        print(theta)
                        # check init pos
                        if abs(x-x_init) < 0.01 and abs(y-y_init) < 0.01 and abs(theta-theta_init) < 1:
                            robot_in_pos = True
                            #sleep(2)
                        else:
                            robot_in_pos = False
                            
                    # First acion
                    elif not first_action_taken:
                        print("get 1st action")
                        ( lidar, angles ) = lidarScan(msgScan)
                        ( state_ind, x1, x2 ,x3 ,x4 ) = scanDiscretization(state_space, lidar)
                        crash = checkCrash(lidar)

                        if EXPLORATION_FUNCTION == 1 :
                            ( action, status_strat ) = softMaxSelection(Q_table, state_ind, actions, T)
                            print('action is ', action)
                        else:
                            ( action, status_strat ) = epsiloGreedyExploration(Q_table, state_ind, actions, T)

                        status_rda = robotDoAction(self.velPub, action)

                        prev_lidar = lidar
                        prev_action = action
                        prev_state_ind = state_ind

                        first_action_taken = True

                        if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                            print('\r\n', status_strat, '\r\n')
                            log_sim_info.write('\r\n'+status_strat+'\r\n')

                        if not status_rda == 'robotDoAction => OK':
                            print('\r\n', status_rda, '\r\n')
                            log_sim_info.write('\r\n'+status_rda+'\r\n')
                    
                    # Rest of the algorithm
                    else:
                        print("episodes", episode)
                        ( lidar, angles ) = lidarScan(msgScan)
                        ( state_ind, x1, x2 ,x3 ,x4 ) = scanDiscretization(state_space, lidar)
                        crash = checkCrash(lidar)

                        ( reward, terminal_state ) = getReward(action, prev_action, lidar, prev_lidar, crash)

                        ( Q_table, status_uqt ) = updateQTable(Q_table, prev_state_ind, action, reward, state_ind, alpha, gamma)

                        if EXPLORATION_FUNCTION == 1:
                            ( action, status_strat ) = softMaxSelection(Q_table, state_ind, actions, T)
                        else:
                            ( action, status_strat ) = epsiloGreedyExploration(Q_table, state_ind, actions, T)

                        status_rda = robotDoAction(self.velPub, action)

                        if not status_uqt == 'updateQTable => OK':
                            print('\r\n', status_uqt, '\r\n')
                            log_sim_info.write('\r\n'+status_uqt+'\r\n')
                        if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                            print('\r\n', status_strat, '\r\n')
                            log_sim_info.write('\r\n'+status_strat+'\r\n')
                        if not status_rda == 'robotDoAction => OK':
                            print('\r\n', status_rda, '\r\n')
                            log_sim_info.write('\r\n'+status_rda+'\r\n')

                        ep_reward = ep_reward + reward
                        ep_reward_arr = np.append(ep_reward_arr, reward)
                        prev_lidar = lidar
                        prev_action = action
                        prev_state_ind = state_ind


def main(args=None):
    rclpy.init(args=args)

    reinforcement_node = LearningNode()

    rclpy.spin(reinforcement_node)

    if reinforcement_node.callShutdown == True:
        reinforcement_node.destroy_node()

    rclpy.shutdown()