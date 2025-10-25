#!/usr/bin/env python3
import time
import board
import busio
import adafruit_bno055

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

print("Move the BNO055 around in a figure-8 and rotation pattern.")
print("Calibration status: SYS | GYRO | ACCEL | MAG")

try:
    while True:
        cal_status = sensor.calibration_status
        sys, gyro, accel, mag = cal_status
        print(f"Calibration: {sys} | {gyro} | {accel} | {mag}", end="\r")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nCalibration ended.")
