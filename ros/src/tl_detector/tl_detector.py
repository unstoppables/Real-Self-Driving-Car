#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point, PointStamped, TwistStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import math
import cv2
import yaml
import numpy as np
from scipy import spatial

STATE_COUNT_THRESHOLD = 2

MAX_DECEL = 1.


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.KDTree = None
        self.camera_image = None
        self.lights = []
        self.stop_lines = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        # [alexm]NOTE: we should rely on this topic's data except state of the light
        # [alexm]NOTE: according to this: https://carnd.slack.com/messages/C6NVDVAQ3/convo/C6NVDVAQ3-1504625321.000063/
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)
        rospy.Subscriber('/next_wp', Int32, self.next_wp_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)
        config_string = rospy.get_param("/traffic_light_config")
        self.initialized = False
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()

        self.listener = tf.TransformListener()

        self.current_velocity = None
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.next_wp = None

        self.initialized = True

        rospy.spin()

    def current_velocity_cb(self, msg):
        self.current_velocity = msg

    def next_wp_cb(self, val):
        self.next_wp = val.data

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, lane):
        if self.waypoints != lane.waypoints:
            data = []
            for wp in lane.waypoints:
                data.append((wp.pose.pose.position.x, wp.pose.pose.position.y))
            self.KDTree = spatial.KDTree(data)
            self.waypoints = lane.waypoints
            self.stop_lines = None

    def find_stop_line_position(self, light):
        """Finds stop line position from config corresponding to given light 
        Args:
            msg (Image): image from car-mounted camera
        """

        stop_line_positions = self.config['stop_line_positions']
        min_distance = 100000
        result = None
        light_pos = light.pose.pose.position
        for pos in stop_line_positions:
            distance = self.euclidean_distance_2d(pos, light_pos)
            if (distance < min_distance):
                min_distance = distance
                result = pos
        return result

    def traffic_cb(self, msg):
        if not self.stop_lines and self.KDTree:
            stop_lines = []
            for light in msg.lights:
                # find corresponding stop line position from config
                stop_line_pos = self.find_stop_line_position(light)
                # find corresponding waypoint indicex
                closest_index = self.KDTree.query(np.array([stop_line_pos]))[1][0]
                closest_wp = self.waypoints[closest_index]
                if not self.is_ahead(closest_wp.pose.pose, stop_line_pos):
                    closest_index = max(closest_index - 1, 0)
                # add index to list
                stop_lines.append(closest_index)
            # update lights and stop line waypoint indices
            self.lights = msg.lights
            self.stop_lines = stop_lines

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        if (not self.initialized):
            return

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state in [TrafficLight.RED, TrafficLight.YELLOW] else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def project_to_image_plane(self, stamped_world_point):
        """Project point from 3D world coordinates to 2D camera image location
        Args:
            point_in_world (Point): 3D location of a point in the world
        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image
        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        stamped_world_point.header.stamp = rospy.Time.now()
        base_point = None

        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                                           "/world", now, rospy.Duration(1.0))
            base_point = self.listener.transformPoint("/base_link", stamped_world_point);

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        if not base_point:
            return (0, 0)

        base_point = base_point.point

        print (base_point)

        cx = image_width / 2
        cy = image_height / 2
        # Ugly workaround wrong camera settings
        if fx < 10.:
            fx = 2344.
            fy = 2552.
            cy = image_height
            base_point.z -= 1

        cam_matrix = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]])
        obj_points = np.array([[- base_point.y, - base_point.z, base_point.x]]).astype(np.float32)
        result, _ = cv2.projectPoints(obj_points, (0, 0, 0), (0, 0, 0), cam_matrix, None)

        # print(result)
        cx = image_width / 2
        cy = image_height

        x = int(result[0, 0, 0])
        y = int(result[0, 0, 1])
        print(x, y)

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # TODO use light location to zoom in on traffic light in image
        # Projection wont work cuz of absent base_link->world transform on site

        # tl_point = PointStamped()
        # tl_point.header = light.pose.header
        # tl_point.point = Point()
        # tl_point.point.x = light.pose.pose.position.x
        # tl_point.point.y = light.pose.pose.position.y
        # tl_point.point.z = light.pose.pose.position.z
        # x, y = self.project_to_image_plane(tl_point)

        # Get classification
        state = self.light_classifier.get_classification(cv_image)
        if state == TrafficLight.UNKNOWN and self.last_state:
            state = self.last_state

        if state == TrafficLight.YELLOW:
            print "Yellow"
        elif state == TrafficLight.GREEN:
            print "Green"
        elif state == TrafficLight.RED:
            print "Red"
        else:
            print "Unknown"

        return state

    def stop_path(self, twist_stamped, decel):
        v = twist_stamped.twist.linear.x
        return 0.5 * v * v / decel

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = None
        light_wp = -1
        min_distance = 0.

        if (self.waypoints and self.next_wp and self.stop_lines):
            next_wp = self.waypoints[min(self.next_wp, len(self.waypoints) - 1)]
            search_distance = self.stop_path(next_wp.twist, MAX_DECEL)
            min_distance = search_distance
            for i in range(len(self.stop_lines)):
                stop_line_wp_index = self.stop_lines[i]
                if stop_line_wp_index >= self.next_wp:
                    stop_line_wp = self.waypoints[stop_line_wp_index]
                    distance = self.euclidean_distance_2d(next_wp.pose.pose.position, stop_line_wp.pose.pose.position)
                    if (distance < min_distance):
                        light_wp = stop_line_wp_index
                        light = self.lights[i]
                        min_distance = distance
                    #        print('n_wp:{}; l_wp:{}'.format(self.next_wp, light_wp))
        if light_wp > -1:
            state = self.get_light_state(light)
            if self.current_velocity and state == TrafficLight.YELLOW \
                    and (min_distance < self.stop_path(self.current_velocity, MAX_DECEL)):
                print "Too close Yellow - IGNORING"
                return -1, TrafficLight.UNKNOWN
            else:
                return light_wp, state
        return -1, TrafficLight.UNKNOWN

    def is_ahead(self, origin_pose, test_position):
        """Determines if test position is ahead of origin_pose
        Args:
            origin_pose - geometry_msgs.msg.Pose instance
            test_position - geometry_msgs.msg.Point instance, or tuple (x,y)
        Returns:
            bool: True iif test_position is ahead of origin_pose
        """
        test_x = self.get_x(test_position)
        test_y = self.get_y(test_position)

        orig_posit = origin_pose.position
        orig_orient = origin_pose.orientation
        quaternion = (orig_orient.x, orig_orient.y, orig_orient.z, orig_orient.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        orig_x = ((test_x - orig_posit.x) * math.cos(yaw) \
                  + (test_y - orig_posit.y) * math.sin(yaw))
        return orig_x > 0.

    def euclidean_distance_2d(self, position1, position2):
        """Finds distance between two position1 and position2
        Args:
            position1, position2 - geometry_msgs.msg.Point instance, or tuple (x,y)
        Returns:
            double: distance
        """
        a_x = self.get_x(position1)
        a_y = self.get_y(position1)
        b_x = self.get_x(position2)
        b_y = self.get_y(position2)
        return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)

    def get_x(self, pos):
        return pos.x if isinstance(pos, Point) else pos[0]

    def get_y(self, pos):
        return pos.y if isinstance(pos, Point) else pos[1]


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
	rospy.logerr('Could not start traffic node.')
