#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import time

"""The goal of this node is to:
1) find the closest path waypoint to our current vehicule position (car_position)
2) find the closest red light's stop line to "car_position" (closest_tl)
3) find the closest path waypoint to "closest_tl"
4) return its id and its color code if it does exists

How to solve the complexity problem of calculating distances over large
set of points in a reasonable amount of time. Here we used the so called "Kd Tree algorithm"
(k-dimensional tree) which is one solution to the Nearest neighbor search problem (NNS)
which is the optimization problem of finding the point in a given set that is closest
(or most similar) to a given point https://en.wikipedia.org/wiki/Nearest_neighbor_search

note: this algorithm work for K dimension but we are using it for 2 here.

This method implies to first build a tree of our set of points using recursivity,
and then search in that tree the closest point to our target point.

Complexity:
Given a set S of points in a space M
The naive method has a running time of O(dN) where N is the cardinality of S and d is the dimensionality of M
The K-d Tree method is O(log n) complex

A good youtube video about K-d trees https://youtu.be/u4M5rRYwRHs
"""

STATE_COUNT_THRESHOLD = 3


def distance(point1, point2):
    """Caculate euclidian distance between two point in a plan.

    Args:
        point1: First tuple (x1, y1)
        point2: Second tuple (x2, y2)

    Returns:
        A float representing the distance between the two points
    """
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    dx = x1 - x2
    dy = y1 - y2

    return math.sqrt(dx * dx + dy * dy)


k = 2


def build_kdtree(points, depth=0):
    """Build the Kd_tree of a list of size n.

    Args:
        points: Sub sample set of points (x1, y1)

    Returns:
        A float representing the distance between the two points
    """
    n = len(points) - 1

    if n <= 0:
        return None

    axis = depth % k

    sorted_points = sorted(points, key=lambda point: point[axis])

    return {
        'point': sorted_points[n / 2],
        'left': build_kdtree(sorted_points[:n / 2], depth + 1),
        'right': build_kdtree(sorted_points[n / 2 + 1:], depth + 1)
    }


def closer_distance(pivot, p1, p2):
    """Determine whether right side point or left
    side point is closest to the pivot.

    Args:
        pivot: Pivot coordinate (xp, yp)
        p1: First point (x1, y1)
        p2: Second point (x2, y2)

    Returns:
        A tuple which is the closest point to the pivot
    """
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    d1 = distance(pivot, p1)
    d2 = distance(pivot, p2)

    if d1 < d2:
        return p1
    else:
        return p2


def kdtree_closest_point(root, point, depth=0):
    """Determine whether right side point or left
    side point is closest to the pivot.

    Args:
        root: The K-d Tree for the set of point which is a deep nested list
        point: The point for which we want to find the closest (xp, yp)
        depth: The depth of the research in the tree which start at 0
        and increment by 1 at each recursion

    Returns:
        Returns the closest point of the set (xcp, ycp)
    """
    if root is None:
        return None

    axis = depth % k

    next_branch = None
    opposite_branch = None

    if point[axis] < root['point'][axis]:
        next_branch = root['left']
        opposite_branch = root['right']
    else:
        next_branch = root['right']
        opposite_branch = root['left']

    best = closer_distance(point,
                           kdtree_closest_point(next_branch,
                                                point,
                                                depth + 1),
                           root['point'])

    if distance(point, best) > abs(point[axis] - root['point'][axis]):
        best = closer_distance(point,
                               kdtree_closest_point(opposite_branch,
                                                    point,
                                                    depth + 1),
                               best)

    return best


class TLDetector(object):

    def __init__(self):
        rospy.init_node('tl_detector')
        self.init_ok = False
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_list = None
        self.kdtree = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights',
                                TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher(
            '/traffic_waypoint', Int32, queue_size=1)

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0


        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        self.init_ok = True



        rospy.spin()


    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        x = list()
        y = list()
        ind = list()

        for i, waypoint in enumerate(waypoints.waypoints):
            x.append(waypoint.pose.pose.position.x)
            y.append(waypoint.pose.pose.position.y)
            ind.append(i)

        self.waypoints_list = list(zip(x, y, ind))
        self.kdtree = build_kdtree(self.waypoints_list)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
        of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        # rospy.logerr("light_wp")
        # rospy.logerr(light_wp)

        # self.upcoming_red_light_pub.publish(Int32(light_wp))

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
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.kdtree:
            return kdtree_closest_point(self.kdtree,
                                        (pose.position.x,
                                         pose.position.y))
        else:
            return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color
            (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #cv2.imshow('image',cv_image)
        #cv2.waitKey(0)

        # Get classification
        if self.init_ok == True:
            return self.light_classifier.get_classification(cv_image)
        else:
            return TrafficLight.UNKNOWN



    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming
            stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color
            (specified in styx_msgs/TrafficLight)

        """
        light = None

        car_position = None

        light_wp = None

        # List of positions that correspond to the line
        # to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        # TODO find the closest visible traffic light (if one exists)

        # rospy.logerr(self.waypoints_list[5000:6000])

        closest_upcoming_tl = None

        if car_position:

            car_index = car_position[2]
            next_car_index = car_index + 1
            next_car_position = self.waypoints_list[next_car_index]

            for stop_line_position in stop_line_positions:
                # Add None to the last element to not block
                stop_line_index = kdtree_closest_point(
                    self.kdtree, stop_line_position)[2]
                if stop_line_index > car_index:
                    light_wp = stop_line_index
                    light = True
                    break

            # time.sleep(1000)

            # rospy.logerr(car_position)
            # rospy.logerr(upcoming_stop_line_positions)

        #    tl_kdtree = build_kdtree(stop_line_positions)
        #    closest_upcoming_tl = kdtree_closest_point(tl_kdtree, car_position)
#
        #if closest_upcoming_tl:
        #    light = True
        #    light_wp = kdtree_closest_point(
        #        self.kdtree, closest_upcoming_tl)[2]

        # rospy.logerr(light_wp)

        if light:
            state = 1
            state = self.get_light_state(light)
            # rospy.logerr("state")
            # rospy.logerr(state)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
