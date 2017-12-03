#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Bool, Int32, Float32
import copy
import math
import time
import tf

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

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


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


class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint
        # and /obstacle_waypoint below
        # commented the below two subscribers for the first phase of project
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', PoseStamped, self.obstacle_cb)

        # Adding Subscriber to get current vehicle (linear) velocity to
        # populate the final_waypoints
        rospy.Subscriber('/current_velocity', TwistStamped,
                         self.current_velocity_cb)

        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.track = None
        self.tracklen = None

        self.pose = None
        self.pose_position = None
        self.current_velocity = None
        self.red_light_waypoint = None

        self.kdtree = None

        # Get system wide parameters
        self.MAX_ACCEL = rospy.get_param('/dbw_node/accel_limit', 1.0)  # m/s2
        self.MAX_DECEL = rospy.get_param('/dbw_node/decel_limit', -5.0)  # m/s2
        self.MAX_SPEED = rospy.get_param(
            '/waypoint_loader/velocity') / 3.6  # kmph->m/s

        self.loop()


    def loop(self):
        rate = rospy.Rate(50)  # 10Hz
        while not rospy.is_shutdown():
            self.publish_final_wps()
            rate.sleep()

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x
        # rospy.logerr("velocity provided")
        # rospy.logerr(msg.twist.linear.x)

    def pose_cb(self, msg):
        self.pose_position = msg.pose.position
        self.pose = msg.pose

    def waypoints_cb(self, waypoints):
        self.track = waypoints
        self.tracklen = len(self.track.waypoints)

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
        self.red_light_waypoint = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it
        # later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0

        def dl(a, b):
            return math.sqrt((a.x - b.x)**2 +
                             (a.y - b.y)**2 +
                             (a.z - b.z)**2)

        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position,
                       waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def find_closest_wp(self):
        min_dist = float("inf")
        wp_i = -1

        def dl(a, b):
            return math.sqrt((a.x - b.x) ** 2 +
                             (a.y - b.y) ** 2 +
                             (a.z - b.z) ** 2)

        for i in range(self.tracklen):
            d = dl(self.pose_position,
                   self.track.waypoints[i].pose.pose.position)
            if d < min_dist:
                min_dist = d
                wp_i = i
        return wp_i

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
                                        (pose.x,
                                         pose.y))
        else:
            return 0

    def dll(self,a, b):
        return math.sqrt((a.x - b.x) ** 2 +
                         (a.y - b.y) ** 2 +
                         (a.z - b.z) ** 2)

    def accelerate_or_keep_velocity(self, ci):

        def dl(a, b):
            return math.sqrt((a.x - b.x) ** 2 +
                            (a.y - b.y) ** 2 +
                            (a.z - b.z) ** 2)
        lane = Lane()
        lane.header.stamp = rospy.Time.now()

        i = 0
        v0 = self.current_velocity
        vi = v0
        a = self.MAX_ACCEL
        while i < LOOKAHEAD_WPS:
            if i:
                d = dl(self.track.waypoints[(ci + i - 1) % self.tracklen].pose.pose.position, self.track.waypoints[(ci + i) % self.tracklen].pose.pose.position)
            else:
                d = dl(self.pose_position, self.track.waypoints[(ci + i) % self.tracklen].pose.pose.position)

            if lane.waypoints:
                vi = lane.waypoints[-1].twist.twist.linear.x

            vi = vi + (a * (d / vi))
            if vi > self.MAX_SPEED:
                vi = self.MAX_SPEED
            current_wp = copy.deepcopy(
                self.track.waypoints[(ci + i) % self.tracklen])
            current_wp.twist.twist.linear.x = vi
            lane.waypoints.append(current_wp)
            i += 1

        return lane

    def decelerate_or_stop(self, ci, red_light_index):
        tolerance_distance = 5
        lane = Lane()
        lane.header.stamp = rospy.Time.now()

        i = 0
        v0 = self.current_velocity
        vi = v0
        a = -10.0 #self.MAX_DECEL
        red_light_waypoint = self.track.waypoints[red_light_index]
        while i < LOOKAHEAD_WPS:
            if (i+ci) > red_light_index:
                vi = 0
            else:
                d = self.dll(red_light_waypoint.pose.pose.position, self.track.waypoints[(ci + i) % self.tracklen].pose.pose.position)
                d = max(0, (d-tolerance_distance))
                vi = math.sqrt(2*a*d)
                if vi < 0:
                    vi = 0.0
            current_wp = copy.deepcopy(self.track.waypoints[(ci + i) % self.tracklen])
            current_wp.twist.twist.linear.x = vi
            lane.waypoints.append(current_wp)
            i += 1

        return lane



    def publish_final_wps(self):

        def get_next_index(car_position):
            """Take the car position index one the waypoint
            and determine wether this point is behind or ahead
            of the car. If it's behind return the next one.
            If it's ahead keep it
            Args:
                car_position (x, y, i): related car waypoint
            Returns:
                int: index of the closest waypoint ahead of the car
            """
            next_index = False

            index_car_position = car_position[2]
            next_index_car_position = index_car_position + 1
            next_car_position = self.waypoints_list[
                next_index_car_position]

            if next_car_position[1] == car_position[1]:
                if car_position[1] < vehicle_posy < next_car_position[1]:
                    next_index = True
                    # rospy.logerr("check_point1")

            elif next_car_position[0] == car_position[0]:
                if car_position[0] < vehicle_posx < next_car_position[0]:
                    next_index = True
                    # rospy.logerr("check_point2")

            else:
                coeff1 = ((next_car_position[1] - car_position[1]) /
                          (next_car_position[0] - car_position[0]))

                coeff2 = -(1 / coeff1)

                b_low = car_position[1] - (coeff2 * car_position[0])
                b_high = next_car_position[1] - (coeff2 * next_car_position[0])

                x_b_p = (self.pose_position.y - b_low) / coeff2
                y_b_p = (coeff2 * self.pose_position.x) + b_low

                x_b_np = (self.pose_position.y - b_high) / coeff2
                y_b_np = (coeff2 * self.pose_position.x) + b_high

                if x_b_p < vehicle_posx < x_b_np and \
                        y_b_p < vehicle_posy < y_b_np:
                    next_index = True

                if next_index:
                    index_car_position = car_position[2] + 1
                    next_index_car_position = index_car_position + 1
                    next_car_position = self.waypoints_list[
                        next_index_car_position]

            return index_car_position


        if self.track and self.pose_position and self.current_velocity:

            # find closest waypoint to the vehicle on the track
            closest_wp_index = self.find_closest_wp()
            ci = closest_wp_index
            #But this index might be some times before or after the current car position
            #If this position is behind the car then we might need to increment the ind
            # Approach is same as in Path Planning Project
            # https://github.com/vkrishnam/SDCND_PathPlanning/blob/51876c3ee62703d88877e3a706943fd2ad510c33/src/main.cpp#L66
            map_x = self.track.waypoints[closest_wp_index].pose.pose.position.x
            map_y = self.track.waypoints[closest_wp_index].pose.pose.position.y
            car_heading_direction_wrt_closest_wp = math.atan2((map_y - self.pose.position.y), (map_x - self.pose.position.x))
            # Quaternion Info in ROS
            # http://docs.ros.org/api/geometry_msgs/html/msg/Quaternion.html
            # http://ros-robotics.blogspot.in/2015/04/getting-roll-pitch-and-yaw-from.html
            # Quaternion to Euler angles
            # http://nullege.com/codes/search/tf.transformations.euler_from_quaternion
            # https://answers.ros.org/question/69754/quaternion-transformations-in-python/
            quaternion = (self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w)
            pitch, roll, yaw = tf.transformations.euler_from_quaternion(quaternion)
            angle = abs(yaw - car_heading_direction_wrt_closest_wp)
            if angle > (math.pi / 4): #Greater than 45 degress then         
                ci += 1



            #car_position = self.get_closest_waypoint(self.pose_position)
            #
            #ci = 0
            #vehicle_posx = self.pose_position.x
            #vehicle_posy = self.pose_position.y
            ## This get_next_index() fucntion is faulty to the extent that input argument is integer
            ## Inside the function it dereference as if it is an array
            #index_car_position = get_next_index(car_position)
            #index_car_position = car_position
            #ci = index_car_position

            if self.red_light_waypoint is None or self.red_light_waypoint < 0:
                lane = self.accelerate_or_keep_velocity(ci)
            else:
                lane = self.decelerate_or_stop(ci, self.red_light_waypoint)
            # Now publish this list
            self.final_waypoints_pub.publish(lane)
            return


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
