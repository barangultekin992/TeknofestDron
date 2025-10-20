import rclpy
from rclpy.node import Node
from pymavlink import mavutil
from main_msgs.srv import Gimbal,Arm,Waypoint,Takeoff,ModeSwitch
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO    
import math
import time
import os

class Copter(Node):
    def __init__(self):
        super().__init__("mavlink_copter")
    
    #obje tanımlama
        self._infer_model = YOLO("/home/baran/ardu_ws/src/main/inference_models/best.pt")
    
    #dron bağlantısını başlat ve ekrana yazdır
        self.drone_connection = mavutil.mavlink_connection("udpin:localhost:14550")
        self.drone_connection.wait_heartbeat(blocking=True)
        self.get_logger().info("Heartbeat from system (system %u component %u)" % (self.drone_connection.target_system, self.drone_connection.target_component))

    #servisler
        self._service_arm = self.create_service(Arm,"mavlink_copter/arming_mav",self.arm)
        self._service_switch_mode = self.create_service(ModeSwitch,"mavlink_copter/switch_mode",self.switch_mode)
        self._service_takeoff = self.create_service(Takeoff,"mavlink_copter/takeoff",self.takeoff)

        self._service_move_ned = self.create_service(Waypoint,"mavlink_copter/move_velocity_ned",self.move_velocity_ned)
        self._service_move_body = self.create_service(Waypoint,"mavlink_copter/move_velocity_body",self.move_velocity_body)
        self._service_move_ned_pos = self.create_service(Waypoint,"mavlink_copter/move_position_ned",self.move_position_ned)
        self._service_move_body_pos = self.create_service(Waypoint,"mavlink_copter/move_position_body",self.move_position_body)
        self._service_gimbal_cmd = self.create_service(Gimbal,"mavlink_copter/gimbal_cmd",self.gimbal_cmd)
        self._service_gimbal_cmd = self.create_service(Gimbal,"mavlink_copter/gimbal_cmd_rate",self.gimbal_cmd_rate)
        self._service_gimbal_set_mode = self.create_service(ModeSwitch,"mavlink_copter/gimbal_mode",self.gimbal_set_mode)
    
    #topicler
        self._publisher_altitude = self.create_publisher(Float64,"mavlink_copter/altitude",10)
        self._publisher_yaw = self.create_publisher(Float64,"mavlink_copter/yaw",10)
        self.create_timer(0.5,self.altitude_pub)
        

    def gimbal_set_mode(self,request,response):
        self.drone_connection.mav.command_long_send(self.drone_connection.target_system,
                                                    self.drone_connection.target_component,
                                                    205,
                                                    0,0,0,0,0,0,0,
                                                    request.mode)
        hb = self.drone_connection.recv_match(type='COMMAND_ACK', blocking=True)
        if hb and hb.command == mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL:
            response.status = (hb.result == mavutil.mavlink.MAV_RESULT_ACCEPTED)
        return response
    def gimbal_cmd(self,request,response):
        """
        yaw positif saat yönünde
        pitch positif yukari doğru
        """
        yaw = math.degrees(request.yaw)
        pitch = 90 - math.degrees(request.pitch)

        self.drone_connection.mav.command_long_send(self.drone_connection.target_system,
                                                    self.drone_connection.target_component,
                                                    1000,0,
                                                    pitch, #pitch
                                                    yaw,#yaw
                                                    float("NaN"), #pitch rate
                                                    float("NaN"), #yaw rate
                                                    0, #Flag (0=Yaw is body-frame/follow, 16=Yaw is earth-frame/lock)
                                                    0,
                                                    0, # gimbal id 0 is primary
        )
        
        hb = self.drone_connection.recv_match(type='COMMAND_ACK', blocking=True)
        if hb and hb.command == mavutil.mavlink.MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW:
            response.status = (hb.result == mavutil.mavlink.MAV_RESULT_ACCEPTED)
        return response

    def gimbal_cmd_rate(self,request,response):
        """
        yaw positif saat yönünde
        pitch positif yukari doğru
        """
        yaw_rate = math.degrees(request.yaw_rate)
        pitch_rate = math.degrees(request.pitch_rate)
        start = time.time()
        while time.time() - start < request.duration:

            self.drone_connection.mav.command_long_send(self.drone_connection.target_system,
                                                        self.drone_connection.target_component,
                                                        1000,0,
                                                        float("NaN"), #pitch
                                                        float("NaN"),#yaw
                                                        pitch_rate, #pitch rate
                                                        yaw_rate, #yaw rate
                                                        0, #Flag (0=Yaw is body-frame/follow, 16=Yaw is earth-frame/lock)
                                                        0,
                                                        0, # gimbal id 0 is primary
            )
        
        self.drone_connection.mav.command_long_send(self.drone_connection.target_system,
                                                        self.drone_connection.target_component,
                                                        1000,0,
                                                        float("NaN"), #pitch
                                                        float("NaN"),#yaw
                                                        0, #pitch rate
                                                        0, #yaw rate
                                                        0, #Flag (0=Yaw is body-frame/follow, 16=Yaw is earth-frame/lock)
                                                        0,
                                                        0, # gimbal id 0 is primary
            )
        hb = self.drone_connection.recv_match(type='COMMAND_ACK', blocking=True)
        if hb and hb.command == mavutil.mavlink.MAV_CMD_DO_GIMBAL_MANAGER_PITCHYAW:
            response.status = (hb.result == mavutil.mavlink.MAV_RESULT_ACCEPTED)
        return response
                                                    

    

    def altitude_pub(self):
        pub_alt = Float64()
        yaw = Float64()
        msg = self.drone_connection.recv_match(type="GLOBAL_POSITION_INT",blocking = True)
        if msg:
            alt_relative = msg.relative_alt / 1000.0  
            pub_alt.data = alt_relative
            yaw_msg = msg.hdg / 100
            yaw.data = yaw_msg
            self._publisher_altitude.publish(pub_alt)
            self._publisher_yaw.publish(yaw)

    def arm(self,request,response):
        self.drone_connection.mav.command_long_send(self.drone_connection.target_system,
                                                    self.drone_connection.target_component,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,request.arm,0,0,0,0,0,0)
        hb = self.drone_connection.recv_match(type='COMMAND_ACK', blocking=True)
        if hb and hb.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            response.status = (hb.result == mavutil.mavlink.MAV_RESULT_ACCEPTED)
        
        return response
    
    def switch_mode(self,request,response):
        mode_id = request.mode
        #mode_id = self.drone_connection.mode_mapping().get(request.mode)
        hb = self.drone_connection.recv_match(type='HEARTBEAT', blocking=True)
        first_mode = hb.custom_mode
        if mode_id is None:
            self.get_logger().error("mode unavailable {mode}")
            response.status = False
            return response
        self.drone_connection.set_mode_apm(mode=mode_id)

        for _ in range(10):
            hb = self.drone_connection.recv_match(type='HEARTBEAT', blocking=True)
            if hb.custom_mode== mode_id:
                response.status = True
                response.curr_mode = request.mode
                return response
        response.status = False
        response.curr_mode = first_mode
        return response
    
    def takeoff(self,request,response):
        
        takeoff_alt = request.alt
        self.drone_connection.mav.command_long_send(self.drone_connection.target_system,
                                                    self.drone_connection.target_component,mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,0,takeoff_alt)
        hb = self.drone_connection.recv_match(type='COMMAND_ACK', blocking=True)
        if hb and hb.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
            response.status = hb.result == mavutil.mavlink.MAV_RESULT_ACCEPTED

        return response
    
    def move_velocity_ned(self,request,response):
        
        start_time =time.time()
        if request.yaw is None :
            bitmask = 0b0000111111000111
        else:
            bitmask = 0b0000101111000111

        duration = request.sec
        while time.time()-start_time < duration:
            self.drone_connection.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                10,  # Time_boot_ms
                self.drone_connection.target_system,
                self.drone_connection.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,  # Correct frame
                bitmask,
                0, 0, 0,  # Position (ignored)
                request.x, request.y, request.z,  # Velocity in m/s
                0, 0, 0,  # Acceleration (ignored)
                request.yaw if request.yaw is not None else 0, 0  # Yaw and Yaw rate
                )
            )
            time.sleep(1)
        self.drone_connection.mav.send(
        mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            10,
            self.drone_connection.target_system,
            self.drone_connection.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
            0b0000111111000111,
            0, 0, 0,
            0.0, 0.0, 0.0,  # Velocity 0 to stop movement
            0, 0, 0,
            0, 0
            )
        )
        response.status = True
        return response
    
    def move_velocity_body(self,request,response):        
        start_time =time.time()
        
        # Body frame için pozisyon kontrolü - bitmask düzeltildi
        if request.yaw is None:
            bitmask = 0b0000111111000111  # Position control without yaw
        else:
            bitmask = 0b0000101111000111  # Position control with yaw

        duration = request.sec
        while time.time()-start_time < duration:
            self.drone_connection.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                10,  
                self.drone_connection.target_system,
                self.drone_connection.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # Body frame
                bitmask,
                0, 0, 0,
                request.x, request.y, request.z,  # VELOCITY in body frame
                0, 0, 0,  # Acceleration (ignored)
                request.yaw if request.yaw is not None else 0, 0  # Yaw and Yaw rate
                )
            )
            time.sleep(0.1)
        
        # Durdurma mesajı - aynı frame kullanılmalı
        self.drone_connection.mav.send(
        mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            10,
            self.drone_connection.target_system,
            self.drone_connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # Aynı frame kullanılmalı
            0b110111000111,
            0, 0, 0,
            0.0, 0.0, 0.0,  # Velocity 0 to stop movement
            0, 0, 0,
            0, 0
            )
        )   
        response.status=True
        return response
    
    def move_position_body(self,request,response):
        self.drone_connection.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                10,
                self.drone_connection.target_system,
                self.drone_connection.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                0b110111000000,
                request.lat,request.lon,request.alt,
                request.x,request.y,request.z,
                0.,0.,0.,
                0,0
            )
        )
        response.status=True
        return response
    
    def move_position_ned(self,request,response):
        self.drone_connection.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
                10,
                self.drone_connection.target_system,
                self.drone_connection.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
                0b110111000000,
                request.lat,request.lon,request.alt,
                request.x,request.y,request.z,
                0.,0.,0.,
                0,0
            )
        )
        response.status=True
        return response
def main():
    rclpy.init()
    drone = Copter()
    rclpy.spin(drone)

if __name__ == '__main__':
    main()