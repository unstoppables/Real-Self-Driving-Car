#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Bool, Int32, Float32
import copy
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        #  commented the below two subscribers for the first phase of project  
        #rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', PoseStamped, self.obstacle_cb)

        #Adding Subscriber to get current vehicle (linear) velocity to populate the final_waypoints
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.track = None
        self.tracklen = None

        self.pose_position = None
        self.current_velocity = None

        #Get system wide parameters
        self.MAX_ACCEL = rospy.get_param('/dbw_node/accel_limit', 1.0) #m/s2
        self.MAX_DECEL = rospy.get_param('/dbw_node/decel_limit', -5.0) #m/s2
        self.MAX_SPEED = rospy.get_param('/waypoint_loader/velocity')/3.6 #kmph->m/s

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)  #10Hz
        while not rospy.is_shutdown():
            self.publish_final_wps()
            rate.sleep()

    def current_velocity_cb(self, msg):
        self.current_velocity =  msg.twist.linear.x

    def pose_cb(self, msg):
        self.pose_position = msg.pose.position
        
    def waypoints_cb(self, waypoints):
        self.track = waypoints
        self.tracklen = len(self.track.waypoints)
        
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def find_closest_wp(self):
        min_dist = 1000000000000.0
        wp_i = -1
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(self.tracklen):
            d = dl(self.pose_position, self.track.waypoints[i].pose.pose.position)
            if d < min_dist:
                min_dist = d
                wp_i = i
        return wp_i

    def publish_final_wps(self):
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        if self.track == None or self.pose_position == None or self.current_velocity == None:
            return
        
        #find closest waypoint to the vehicle on the track
        closest_wp_index = self.find_closest_wp()
        ci =  closest_wp_index

        lane = Lane()
        lane.header.stamp = rospy.Time.now()
        
        i = 0
        v0 = self.current_velocity
        vi = v0
        a = self.MAX_ACCEL
        while i < LOOKAHEAD_WPS:
            d = dl(self.pose_position, self.track.waypoints[(ci+i)%self.tracklen].pose.pose.position)
            vi = math.sqrt(v0**2.0+2.0*a*d)
            if vi > self.MAX_SPEED:
                vi = self.MAX_SPEED
            current_wp = copy.deepcopy(self.track.waypoints[(ci+1)%self.tracklen])
            current_wp.twist.twist.linear.x = vi		
            lane.waypoints.append(current_wp)
            i += 1

        #Now publish this list
        self.final_waypoints_pub.publish(lane)
        return


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
