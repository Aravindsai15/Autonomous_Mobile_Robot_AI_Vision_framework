#!/usr/bin/env python3
# -*- coding: utf-8 -*-





import rospy
from sensor_msgs.msg import Imu, MagneticField
import smbus2
import numpy as np
import time
from threading import Lock
from collections import deque

class IMUDriverNode:
    def __init__(self):
        # Parameters
        self.i2c_bus = rospy.get_param('~i2c_bus', 1)
        self.i2c_address = rospy.get_param('~i2c_address', 0x28)
        self.frame_id = rospy.get_param('~frame_id', 'imu_link')
        self.publish_rate = rospy.get_param('~publish_rate', 50.0)
        self.max_retry_count = rospy.get_param('~max_retry_count', 5)
        self.health_window_size = rospy.get_param('~health_window_size', 20)
        self.orientation_covariance = rospy.get_param('~orientation_covariance', 0.01)
        self.angular_velocity_covariance = rospy.get_param('~angular_velocity_covariance', 0.01)
        self.linear_acceleration_covariance = rospy.get_param('~linear_acceleration_covariance', 0.01)

        # State
        self.i2c_lock = Lock()
        self.last_valid_time = rospy.Time.now()
        self.error_count = 0
        self.health_monitor = deque(maxlen=self.health_window_size)
        self.current_mode = None

        # IMU init
        if not self.initialize_imu():
            rospy.logfatal("Failed to initialize IMU after retries. Shutting down.")
            rospy.signal_shutdown("IMU init failed")
            return

        # Publishers
        self.imu_pub = rospy.Publisher('imu/data', Imu, queue_size=10)
        self.mag_pub = rospy.Publisher('imu/mag', MagneticField, queue_size=10)
        self.rate = rospy.Rate(self.publish_rate)

    def initialize_imu(self):
        for attempt in range(self.max_retry_count):
            try:
                with self.i2c_lock:
                    self.bus = smbus2.SMBus(self.i2c_bus)
                    chip_id = self.bus.read_byte_data(self.i2c_address, 0x00)
                    if chip_id != 0xA0:
                        raise RuntimeError("Unexpected chip ID: 0x{:02X}".format(chip_id))

                    self.bus.write_byte_data(self.i2c_address, 0x3D, 0x00)  # Config mode
                    time.sleep(0.02)
                    self.bus.write_byte_data(self.i2c_address, 0x3B, 0x00)  # Reset
                    time.sleep(0.1)
                    self.bus.write_byte_data(self.i2c_address, 0x3D, 0x0C)  # NDOF
                    time.sleep(0.2)

                    self.current_mode = "NDOF"
                    rospy.loginfo("BNO055 initialized successfully")
                    return True
            except Exception as e:
                rospy.logerr("Initialization attempt {} failed: {}".format(attempt + 1, str(e)))
                time.sleep(0.5)
        return False

    def read_imu_data(self):
        try:
            with self.i2c_lock:
                timestamp = rospy.Time.now()
                quat = self.bus.read_i2c_block_data(self.i2c_address, 0x20, 8)
                accel = self.bus.read_i2c_block_data(self.i2c_address, 0x08, 6)
                gyro = self.bus.read_i2c_block_data(self.i2c_address, 0x14, 6)
                mag = self.bus.read_i2c_block_data(self.i2c_address, 0x0E, 6)
                calib = self.bus.read_byte_data(self.i2c_address, 0x35)

            q = [self.twos_comp((quat[i+1] << 8) | quat[i], 16) / 16384.0 for i in range(0, 8, 2)]
            a = [self.twos_comp((accel[i+1] << 8) | accel[i], 16) / 100.0 for i in range(0, 6, 2)]
            g = [np.radians(self.twos_comp((gyro[i+1] << 8) | gyro[i], 16) / 16.0) for i in range(0, 6, 2)]
            m = [self.twos_comp((mag[i+1] << 8) | mag[i], 16) / 16.0 for i in range(0, 6, 2)]

            self.validate_data(q, a, g, m)

            return {
                'timestamp': timestamp,
                'quat': q,
                'accel': a,
                'gyro': g,
                'mag': m,
                'calibration': calib
            }

        except Exception as e:
            self.error_count += 1
            self.health_monitor.append(False)
            rospy.logwarn("IMU read failed: {}".format(str(e)))
            return None

    def validate_data(self, quat, accel, gyro, mag):
        if not all(np.isfinite(x) for x in quat + accel + gyro + mag):
            raise ValueError("Non-finite IMU data detected")

        q_norm = np.linalg.norm(quat)
        if not 0.95 < q_norm < 1.05:
            raise ValueError("Invalid quaternion norm: {:.4f}".format(q_norm))

        accel_norm = np.linalg.norm(accel)
        if not 8.0 < accel_norm < 11.0:
            raise ValueError("Unusual acceleration magnitude: {:.2f} m/s²".format(accel_norm))

    def twos_comp(self, val, bits):
        """Compute the 2's complement of a raw IMU value."""
        if val & (1 << (bits - 1)):  # If sign bit is set
            val = val - (1 << bits)   # Compute negative value
        return val

    def create_imu_message(self, data):
        msg = Imu()
        msg.header.stamp = data['timestamp']
        msg.header.frame_id = self.frame_id

        q = data['quat']
        q_norm = np.linalg.norm(q)
        if abs(q_norm - 1.0) > 0.01:
            q = [x / q_norm for x in q]
            rospy.logwarn("Normalized quaternion (was {:.4f})".format(q_norm))

        msg.orientation.w = float(q[0])
        msg.orientation.x = float(q[1])
        msg.orientation.y = float(q[2])
        msg.orientation.z = float(q[3])

        msg.angular_velocity.x = float(data['gyro'][0])
        msg.angular_velocity.y = float(data['gyro'][1])
        msg.angular_velocity.z = float(data['gyro'][2])

        msg.linear_acceleration.x = float(data['accel'][0])
        msg.linear_acceleration.y = float(data['accel'][1])
        msg.linear_acceleration.z = float(data['accel'][2])

        msg.orientation_covariance = [self.orientation_covariance] * 9
        msg.angular_velocity_covariance = [self.angular_velocity_covariance] * 9
        msg.linear_acceleration_covariance = [self.linear_acceleration_covariance] * 9

        return msg

    def create_mag_message(self, data):
        msg = MagneticField()
        msg.header.stamp = data['timestamp']
        msg.header.frame_id = self.frame_id

        msg.magnetic_field.x = float(data['mag'][0])
        msg.magnetic_field.y = float(data['mag'][1])
        msg.magnetic_field.z = float(data['mag'][2])
        msg.magnetic_field_covariance = [0.1] * 9

        return msg

    def update_loop(self):
        data = self.read_imu_data()
        if data is None:
            error_rate = sum(self.health_monitor) / float(len(self.health_monitor)) if self.health_monitor else 0
            if error_rate < 0.5:
                return
            rospy.logerr("High IMU error rate ({:.0%}), attempting recovery...".format(error_rate))
            self.recover_imu()
            return

        self.health_monitor.append(True)
        self.last_valid_time = rospy.Time.now()

        imu_msg = self.create_imu_message(data)
        mag_msg = self.create_mag_message(data)

        self.imu_pub.publish(imu_msg)
        self.mag_pub.publish(mag_msg)

    def recover_imu(self):
        rospy.logwarn("Attempting IMU recovery...")
        try:
            with self.i2c_lock:
                self.bus.write_byte_data(self.i2c_address, 0x3D, 0x00)  # Config mode
                time.sleep(0.1)
                self.bus.write_byte_data(self.i2c_address, 0x3B, 0x00)  # Reset
                time.sleep(0.2)
                self.bus.write_byte_data(self.i2c_address, 0x3D, 0x0C)  # NDOF
                time.sleep(0.3)

            self.current_mode = "NDOF"
            self.health_monitor.clear()
            rospy.loginfo("IMU recovery successful")

        except Exception as e:
            rospy.logfatal("IMU recovery failed: {}".format(str(e)))
            if not self.initialize_imu():
                rospy.logfatal("Re-initialization failed. Shutting down.")
                rospy.signal_shutdown("IMU unrecoverable")

    def run(self):
        while not rospy.is_shutdown():
            self.update_loop()
            self.rate.sleep()

def main():
    rospy.init_node('imu_driver')
    node = IMUDriverNode()
    node.run()

if __name__ == '__main__':
    main()
