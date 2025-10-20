
import rclpy
import rclpy.duration
from rclpy.node import Node
from ardupilot_msgs.srv import ArmMotors
from ardupilot_msgs.srv import ModeSwitch
from ardupilot_msgs.srv import Takeoff
from ardupilot_msgs.msg import GlobalPosition
from geometry_msgs.msg import TwistStamped
import time

class Copter(Node):
    def __init__(self):
        super().__init__("Copter")
        #arm servisini tanımla
        self._client_arm = self.create_client(ArmMotors,"/ap/arm_motors")
        while not self._client_arm.wait_for_service(10):
            self.get_logger().info("Motor arm etme servisi bekleniyor")
        
        #Uçuş modu değiştirme servisini tanımla
        self._client_mode_switch = self.create_client(ModeSwitch,"/ap/mode_switch")
        while not self._client_mode_switch.wait_for_service(10):
            self.get_logger().info("Uçuş modu değiştirme servisi bekleniyor")
        
        #kalkış servisini tanımla
        self._client_takeoff = self.create_client(Takeoff,"/ap/experimental/takeoff")
        while not self._client_takeoff.wait_for_service(10):
            self.get_logger().info("kalkış servisi bekleniyor")
        self._publisher_move = self.create_publisher(GlobalPosition,"/ap/cmd_gps_pose",10)
        self.msg = GlobalPosition()
        self.msg.velocity.linear.x=0.0
        self.msg.velocity.linear.y=0.0
        self.msg.velocity.linear.z=0.0
        self.msg.coordinate_frame=5
        self.timer = self.create_timer(0.1,self.publish)
    #request(istek) modüllerini yaz
    def mode_switch(self,mode):
        self.req_mode_switch = ModeSwitch.Request()
        self.req_mode_switch.mode=mode
        future_arm=self._client_mode_switch.call_async(self.req_mode_switch)
        rclpy.spin_until_future_complete(self,future=future_arm)
        return future_arm.result()
    
    def switch_mode_with_timeout(self, desired_mode: int, timeout: rclpy.duration.Duration):
        is_in_desired_mode = False
        start = self.get_clock().now()
        while not is_in_desired_mode and self.get_clock().now() - start < timeout:
            result = self.mode_switch(desired_mode)
            # Handle successful switch or the case that the vehicle is already in expected mode
            is_in_desired_mode = result.status or result.curr_mode == desired_mode
            time.sleep(1)

        return is_in_desired_mode
        
    def arm(self,b):
        self.req_arm = ArmMotors.Request()
        self.req_arm.arm = b
        future_mode_switch =self._client_arm.call_async(self.req_arm)
        rclpy.spin_until_future_complete(self,future_mode_switch)
        return future_mode_switch.result()

    def arm_with_timeout(self,b, timeout:rclpy.duration.Duration):
        armed = False
        start = self.get_clock().now()
        while not armed and self.get_clock().now() - start < timeout:
            result=self.arm(b)
            armed = result or result.result == 1
            time.sleep(1)
        return armed
    def takeoff(self,altitude):
        self.req_takeoff = Takeoff.Request()
        self.req_takeoff.alt=altitude
        future_takeoff= self._client_takeoff.call_async(self.req_takeoff)
        rclpy.spin_until_future_complete(self,future_takeoff)
        return future_takeoff.result()
    
    def takeoff_with_timeout(self,altitude,duration:rclpy.duration.Duration):
        takeoff_bool=False
        start = self.get_clock().now()
        while not takeoff_bool and self.get_clock().now() - start < duration:
            result = self.takeoff(altitude)
            takeoff_bool = result
            time.sleep(1)
        return takeoff_bool
    
    def publish(self):
        self._publisher_move.publish(self.msg)

    def move(self,x,y,z,tim):
        self.msg
        self.msg.velocity.linear.y=y
        self.msg.velocity.linear.z=z
        
        start_time = time.time()
        while time.time() - start_time < tim:
            self._publisher_move.publish(self.msg)
            time.sleep(0.1)  # Maintain movement at 10 Hz
        self.msg.velocity.linear.x = 0.0
        self.msg.velocity.linear.y = 0.0
        self.msg.velocity.linear.z = 0.0
        self._publisher_move.publish(self.msg)
    def move_to_coords(self,lat,lon,alt):
        self.msg.latitude=lat
        self.msg.longitude=lon
        self.msg.altitude=alt
        start_time = time.time()
        while time.time() - start_time < 10:
            self._publisher_move.publish(self.msg)
            time.sleep(0.1)  # Maintain movement at 10 Hz

    
    

def main():
    #nodeları tanımla
    rclpy.init()
    drone = Copter()
    
    #Uçuş modunu GUIDED yap
    mode_switched=drone.switch_mode_with_timeout(4,rclpy.duration.Duration(seconds=30))
    drone.get_logger().info("mod değiştirilmesi:{}".format(mode_switched))
    #motorları arm et
    armed = drone.arm_with_timeout(True,rclpy.duration.Duration(seconds=20))
    drone.get_logger().info("Motorlar arm edilmesi:{}".format(armed))
    #10 metreye kalkış gerçekleştir
    takeoff_complete=drone.takeoff_with_timeout(10.5,rclpy.duration.Duration(seconds=30))
    drone.get_logger().info("takeoff:{}".format(takeoff_complete))

    #10 saniye bekle
    time.sleep(10)
    drone.move_to_coords(10.12,0.0,10.5)
    #yere in
    mode_switched=drone.switch_mode_with_timeout(9,rclpy.duration.Duration(seconds=10))
    drone.get_logger().info("mod değiştirilmesi:{}".format(mode_switched))
    #programı durdur
    rclpy.shutdown()
if __name__ == '__main__':
    main()
