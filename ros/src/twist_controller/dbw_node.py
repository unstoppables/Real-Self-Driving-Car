#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.
You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.
One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.
We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.
We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.
Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.
'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        self.decel_limit = rospy.get_param('~decel_limit', -5)
        self.accel_limit = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # TODO: Create `TwistController` object
        # self.controller = TwistController(<Arguments you wish to provide>)

        # define max and mins
        self.max_throttle_percentage = rospy.get_param('~max_throttle_percentage', 0.1) 
        self.max_braking_percentage = rospy.get_param('~max_braking_percentage', -0.1)
        self.min_speed = rospy.get_param('~min_speed', 0.44704)   #one mile per hour
        
        # set up flags
        self.dbw_enabled = False        
        self.rst_flg = True
        self.twist_cmd_recent = None
        self.current_velocity = None
        self.previous_timestamp = rospy.get_time()

        self.controller = Controller(self.wheel_base, self.steer_ratio, self.min_speed, self.accel_limit, self.max_steer_angle, self.vehicle_mass, self.wheel_radius, self.brake_deadband, self.max_throttle_percentage, self.max_braking_percentage, self.max_lat_accel, self.fuel_capacity, self.decel_limit)
        
        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb, queue_size=1)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # 50Hz
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            # throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            # if <dbw is enabled>:
            #   self.publish(throttle, brake, steer)
            if (self.current_velocity is not None) and (self.twist_cmd_recent is not None) and (self.dbw_enabled is True):
                timestamp = rospy.get_time()
                timeDiff = timestamp - self.previous_timestamp
                self.previous_timestamp = timestamp    

                throttle, brake, steering = self.controller.control(self.twist_cmd_recent, self.current_velocity, timeDiff)
                
                if self.dbw_enabled:
                    self.publish(throttle, brake, steering)
                    if steering != 0:
                        rospy.logwarn("steering if dbw_enabled is true: %s", steering)
                    if throttle != 0:
                        rospy.logwarn("throttle %s",throttle)
                    if brake != 0:
                        rospy.logwarn("brake %s", brake)
                
                if self.rst_flg is True:
                    self.controller.pid.reset()
                    self.rst_flg = False
            else:
                self.rst_flg = True

            rate.sleep()

    def dbw_enabled_cb(self, dbw_enabled):
        self.dbw_enabled = bool(dbw_enabled.data)
        rospy.logwarn("DBW_Enabled %s" % dbw_enabled)

    def current_velocity_cb(self, current_velocity):
        self.current_velocity = current_velocity
#        if current_velocity.twist.linear.x != 0:
#            rospy.logwarn("Current Velocity %s" % current_velocity.twist.linear.x)
        
    def twist_cmd_cb(self, twist_cmd):
        self.twist_cmd_recent = twist_cmd
#        if twist_cmd.twist.linear.x != 0:
#            rospy.logwarn("Twist_cmd_Linear_x %s" % twist_cmd.twist.linear.x)
#        if twist_cmd.twist.angular.z != 0:
#            rospy.logwarn("Twist_cmd_Angular_Z %s" % twist_cmd.twist.angular.z)
            
    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
	DBWNode()
