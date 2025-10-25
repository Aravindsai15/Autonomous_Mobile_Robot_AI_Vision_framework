#!/usr/bin/env python3


# import rospy
# import serial
# import time
# from geometry_msgs.msg import Twist

# class CmdVelToSerial:
#     def __init__(self):
#         rospy.init_node('cmd_vel_to_serial', anonymous=True)
        
#         # Get parameters
#         serial_port = rospy.get_param('~serial_port', '/dev/ttyACM0')
#         baud_rate = rospy.get_param('~baud_rate', 115200)
#         timeout = rospy.get_param('~timeout', 0.1)
        
#         # Initialize serial connection with handshake
#         self.ser = None
#         self.connect_serial(serial_port, baud_rate, timeout)
        
#         # Setup subscriber
#         rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
#         rospy.on_shutdown(self.shutdown_hook)
        
#         # Rate limiter (10Hz max)
#         self.rate = rospy.Rate(10)
#         self.last_sent = 0.0

#     def connect_serial(self, port, baud, timeout):
#         """Handle serial connection with retries"""
#         connected = False
#         while not connected and not rospy.is_shutdown():
#             try:
#                 self.ser = serial.Serial(port, baud, timeout=timeout)
#                 time.sleep(2)  # Wait for Arduino reset
#                 self.ser.flushInput()
#                 rospy.loginfo(f"Connected to Arduino at {port}")
#                 connected = True
#             except serial.SerialException as e:
#                 rospy.logwarn(f"Serial connection failed: {e}. Retrying...")
#                 time.sleep(1)
        
#     def cmd_vel_callback(self, msg):
#         """Process Twist messages and send to Arduino"""
#         if self.ser is None or not self.ser.is_open:
#             rospy.logwarn_throttle(5, "Serial port not available")
#             return
            
#         # Rate limiting
#         now = rospy.get_time()
#         if now - self.last_sent < 0.05:  # 20Hz max
#             return
            
#         try:
#             # Format exactly as Arduino expects
#             command = f"{msg.linear.x:.3f} {msg.angular.z:.3f}\n"
#             self.ser.write(command.encode('ascii'))
#             self.last_sent = now
#             rospy.logdebug(f"Sent: {command.strip()}")
#         except serial.SerialException as e:
#             rospy.logwarn(f"Serial write failed: {e}")
#             self.reconnect()

#     def reconnect(self):
#         """Attempt to reconnect to serial"""
#         if self.ser:
#             self.ser.close()
#         self.connect_serial(
#             rospy.get_param('~serial_port', '/dev/ttyACM0'),
#             rospy.get_param('~baud_rate', 115200),
#             rospy.get_param('~timeout', 0.1)
#         )

#     def shutdown_hook(self):
#         """Clean shutdown procedure"""
#         rospy.loginfo("Shutting down - stopping motors")
#         try:
#             if self.ser and self.ser.is_open:
#                 self.ser.write("0.0 0.0\n".encode('ascii'))
#                 self.ser.flush()
#                 self.ser.close()
#         except:
#             pass

# if __name__ == '__main__':
#     try:
#         node = CmdVelToSerial()
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         pass



#!/usr/bin/env python
import rospy
import serial
import time
from geometry_msgs.msg import Twist

class CmdVelToSerial:
    def __init__(self):
        rospy.init_node('cmd_vel_to_serial', anonymous=True)
        
        # Configuration with default values
        self.serial_port = rospy.get_param('~port', '/dev/ttyACM0')
        self.baud_rate = rospy.get_param('~baud', 115200)
        self.timeout = 0.001
        self.min_send_interval = 0.05  # 20Hz max rate
        self.connection_retry_delay = 1.0
        
        # State variables
        self.last_send_time = 0
        self.last_valid_cmd = (0.0, 0.0)
        self.ser = None
        
        # Initialize serial connection
        self.connect_serial()
        
        # ROS setup
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)
        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo("cmd_vel_to_serial node initialized")

    def connect_serial(self):
        """Initialize serial connection with error handling"""
        if self.ser is not None:
            try:
                self.ser.close()
            except:
                pass
        
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                write_timeout=0
            )
            # Alternative to set_buffer_size that works across PySerial versions
            if hasattr(self.ser, 'set_buffer_size'):
                self.ser.set_buffer_size(rx_size=64, tx_size=64)
            time.sleep(2)  # Allow Arduino to reset
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            rospy.loginfo(f"Successfully connected to {self.serial_port} at {self.baud_rate} baud")
            return True
        except (serial.SerialException, OSError) as e:
            rospy.logerr(f"Failed to connect to serial port: {e}")
            self.ser = None
            return False
        except Exception as e:
            rospy.logerr(f"Unexpected connection error: {e}")
            self.ser = None
            return False

    def cmd_vel_callback(self, msg):
        """Process Twist messages with deadzone and rate limiting"""
        try:
            now = time.time()
            if now - self.last_send_time < self.min_send_interval:
                return
                
            # Apply deadzones
            lin_x = 0.0 if abs(msg.linear.x) < 0.01 else msg.linear.x
            ang_z = 0.0 if abs(msg.angular.z) < 0.05 else msg.angular.z
            
            # Skip if no change from last command
            if abs(lin_x - self.last_valid_cmd[0]) < 0.001 and \
               abs(ang_z - self.last_valid_cmd[1]) < 0.001:
                return
                
            self.last_valid_cmd = (lin_x, ang_z)
            
            # Ensure we have a working connection
            if self.ser is None or not self.ser.is_open:
                if not self.connect_serial():
                    rospy.logwarn_throttle(5.0, "Waiting for serial connection...")
                    return
            
            # Send command
            try:
                cmd_str = f"{lin_x:.3f} {ang_z:.3f}\n"
                self.ser.write(cmd_str.encode('ascii'))
                self.ser.flush()
                self.last_send_time = now
                rospy.logdebug(f"Sent: {cmd_str.strip()}")
            except (serial.SerialException, OSError) as e:
                rospy.logwarn(f"Serial write failed: {e}")
                self.ser = None
                self.connect_serial()
                
        except Exception as e:
            rospy.logerr(f"Error in cmd_vel_callback: {e}")
            # Don't crash - keep trying to process messages

    def shutdown_hook(self):
        """Ensure motors stop on shutdown"""
        rospy.loginfo("Stopping motors...")
        if self.ser is not None:
            try:
                self.ser.write("0.0 0.0\n".encode('ascii'))
                self.ser.flush()
                self.ser.close()
            except:
                pass
            self.ser = None

if __name__ == '__main__':
    try:
        node = CmdVelToSerial()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error in main: {e}")