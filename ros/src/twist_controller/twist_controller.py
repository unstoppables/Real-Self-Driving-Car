
from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement

        self.wheel_base= args[0]
        self.steer_ratio= args[1]
        self.min_speed= args[2]
        self.accel_limit= args[3]
        self.max_steer_angle= args[4]
        self.vehicle_mass= args[5]
        self.wheel_radius= args[6]
        self.brake_deadband= args[7]
        self.max_throttle_percentage= args[8]
        self.max_braking_percentage= args[9]
        self.max_lat_accel= args[10]
        self.fuel_capacity= args[11]
        self.decel_limit= args[12]
        
        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, self.min_speed, self.max_lat_accel, self.max_steer_angle)
        self.pid = PID(2.0, 0.4, 0.1, mn=self.max_braking_percentage, mx=self.max_throttle_percentage)

    def control(self, twist_cmd_recent, velocity_now, timeDiff):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        throttle, brake, steering_angle = 0., 0., 0.

        steering_angle = self.yaw_controller.get_steering(twist_cmd_recent.twist.linear.x, twist_cmd_recent.twist.angular.z, velocity_now.twist.linear.x)
        #steering_angle = twist_cmd_recent.twist.angular.z * self.steer_ratio
        acceler = self.pid.step(twist_cmd_recent.twist.linear.x - velocity_now.twist.linear.x, timeDiff)

        if acceler > 0:
            acceler = min(self.accel_limit, acceler)
        else:
            acceler = max(self.decel_limit, acceler)

        if acceler > 0:
            brake = 0.
            throttle = min(1.0, acceler * 4.2) # 4.2 is based on weight of the car and radius of wheel to run it faster
        else:
            if abs(acceler) < self.brake_deadband:
                acceler = 0.
            throttle = 0.
            brake = abs((self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * acceler * self.wheel_radius)
            
        # Return throttle, brake, steer
	return throttle, brake, steering_angle
