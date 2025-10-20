"""
CyberFlight takimi görüntü işleme ve görev kodu. 
bu dosyada çalisma hizini arttirmak icin python listeleri yerine numpy arraylari kullanilmistir.
faydalanilan kaynaklar:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
https://www.youtube.com/@firstprinciplesofcomputerv3258
https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
"""



#kullanılacak kütüphaneleri tanımla
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from main_msgs.srv import Waypoint,Gimbal,Id
from ardupilot_msgs.srv import ArmMotors,ModeSwitch,Takeoff
from std_msgs.msg import Float64
from ultralytics import YOLO
import os
import time
from cv_bridge import CvBridge
import cv2
import subprocess
import math
from math import cos,sin,tan
import numpy as np
from filterpy.kalman import unscented_transform,MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_continuous_white_noise

def fx(X,dt,u):
        # State: [x, vx, y, vy] 
        new_x = X[0] + (X[1] - u[0]) * dt
        new_vel_x = X[1]  
        new_y = X[2] + (X[3] - u[1]) * dt
        new_vel_y = X[3]  
        return np.array([new_x, new_vel_x, new_y, new_vel_y])

def hx(X, rotation_matrix, translation_vector, camera_matrix, dist_coeffs=None):
    """
    Measurement function for UKF - projects 3D world point to image coordinates
    Handles high yaw angles properly with angle normalization
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4,1))  # distorsiyon yok sayılır
    
    # 3D nokta - world coordinates
    object_points = np.array([[X[0], 0.0, X[2]]], dtype=np.float64)  # (N,3)

    # cv2.projectPoints R ve t'yi OpenCV formatında istiyor
    rvec, _ = cv2.Rodrigues(rotation_matrix)  # rot matrisini rot vektörüne çevir
    tvec = translation_vector.reshape(3,1)

    # Projeksiyon
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    u, v = image_points[0][0]
    

    return np.array([u, v])

#kameranın ölçülerini tanımla
FRAME_HEIGHT=480
FRAME_WIDTH=640
FRAME_MIDDLE=np.array([FRAME_WIDTH//2,FRAME_HEIGHT//2])


class Control(Node):
    def __init__(self,p_yaw,p_pitch,i,d,):
        super().__init__("img_process_control")
        
        #PID değerleri
        self.p_yaw=p_yaw
        self.p_pitch = p_pitch
        self.i=i
        self.d=d
        
        self.HFOV = 2.0  # yatay görüş açısı (FOV)
        self.VFOV = self.HFOV * (FRAME_HEIGHT / FRAME_WIDTH)  # dikey görüş açısı (FOV)
        self.focal_x = (FRAME_WIDTH/2) / tan(self.HFOV/2)
        self.focal_y = (FRAME_HEIGHT/2) / tan(self.VFOV/2)
        self.cx,self.cy = FRAME_MIDDLE
        self.get_logger().info(f"focal_x {self.focal_x} focal_y { self.focal_y}")
        
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        #numpy arraylerde append fonksiyonu python listelerindeki gibi olmadığından normal liste
        points = MerweScaledSigmaPoints(4,0.4,2,-1)
        self.ukf = UKF(4,2,0.01,hx,fx,points)

        #ROS'tan opencv ye çevirmek için köprü
        self.bridge=CvBridge()

        self._infer_model = YOLO("/home/baran/ardu_ws/src/main/inference_models/best.pt")
        self.detected_mid_point = None
        self.ID = None
        self.started = False
    #clientlar
        #Body Velocity
        self._client_move_body_vel = self.create_client(Waypoint,"mavlink_copter/move_velocity_body",)
        while not self._client_move_body_vel.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('servis move_body servisi bekleniyor...')
        self.req_body_vel = Waypoint.Request()
        
        #NED Velocity
        self._client_move_ned_vel = self.create_client(Waypoint,"mavlink_copter/move_velocity_ned",)
        while not self._client_move_ned_vel.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('servis move_ned servisi bekleniyor...')
        self.req_ned_vel = Waypoint.Request()
        
        #Body Position
        self._client_move_body_pos = self.create_client(Waypoint,"mavlink_copter/move_position_body",)
        while not self._client_move_body_pos.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('servis move_ned servisi bekleniyor...')
        self.req_body_pos = Waypoint.Request()
        
        #NED Position
        self._client_move_ned_pos = self.create_client(Waypoint,"mavlink_copter/move_position_ned",)
        while not self._client_move_ned_pos.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('servis move_ned servisi bekleniyor...')
        self.req_ned_pos = Waypoint.Request()

        #kalkış clientı        
        self._client_takeoff=self.create_client(Takeoff,"mavlink_copter/takeoff")
        while not self._client_takeoff.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("servis takeoff servisi bekleniyor...")
        self.req_takeoff = Takeoff.Request()

        #uçuş modu değiştirme clientı
        self._client_mode = self.create_client(ModeSwitch,"mavlink_copter/switch_mode")
        while not self._client_takeoff.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("servis mod servisi bekleniyor...")
        self.req_mode = ModeSwitch.Request()
        
        #arm servisi
        self._client_arm = self.create_client(ArmMotors,"mavlink_copter/arming_mav")
        while not self._client_arm.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("servis arm servisi bekleniyor...")
        self.req_arm= ArmMotors.Request()
        
        #gimbal kontrol servisi
        self._client_gimbal_cmd = self.create_client(Gimbal,"mavlink_copter/gimbal_cmd")
        while not self._client_gimbal_cmd.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("servis gimbal servisi bekleniyor...")
        self.req_gimbal_cmd= Gimbal.Request()
    #servisler
        self._service_id = self.create_service(Id,"vision/ID",self.assign_ID)
    #topicler
        self._subscriber_altitude = self.create_subscription(Float64,"/mavlink_copter/altitude",self.assign_altitude,10)
        self._subscriber_cam = self.create_subscription(Image,"/world/map/model/iris/link/pitch_link/sensor/camera/image",self.cam_callback,10)
        self._subscriber_yaw = self.create_subscription(Float64,"/mavlink_copter/yaw",self.assign_yaw,10)
    #sayaç
        self.last_cmd = time.time()
    #methodlar
    def assign_ID(self,request,response): 
        self.ID = request.id
        self.started = False
        response.status = True
        return response
    
    def assign_altitude(self,msg): self.altitude = msg.data - 0.5
    def assign_yaw(self,msg): self.drone_yaw = msg.data 
    
    


    def cvt_to_world(self, altitude, pitch, yaw):
        """
        Convert image coordinates to world coordinates
        x, y: image coordinates
        altitude: drone altitude
        pitch, yaw: gimbal angles
        Returns: (world_x, world_y) in drone body frame
        """
        # Calculate ray direction in world frame
        ray_world = np.array([
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            -math.sin(pitch)
        ])
        
        # Find ground intersection point
        h = altitude
        if abs(ray_world[2]) < 1e-3:
            self.get_logger().warn("Ray is nearly parallel to ground, intersection unreliable.")
            return 0.0, 0.0
        
        t = -h / ray_world[2]
        ground_point = t * ray_world
        
        return ground_point[0], ground_point[1]

    def calculate_gimbal_angles(self, target_x, target_y):
        distance = math.sqrt(target_x**2 + target_y**2)
        
        if distance < 1e-6:
            return 0.0, 0.0
        
        yaw = math.atan2(target_x, target_y)  
        pitch = math.atan2( self.altitude,distance)  # Up-down angle
        
        return yaw, pitch

    def align_axis(self):
        fx= self.focal_x
        fy=self.focal_y
        
        # None kontrolü ekle
        if self.detected_mid_point is None:
            return 0.0, 0.0
            
        detect_x,detect_y = self.detected_mid_point
        dx = detect_x - self.cx
        dy = self.cy - detect_y 
        #normalize et
        x_cam = dx / fx
        y_cam = dy / fy
        z_cam = 1.0
        #vektör haline getir
        r_cam = np.array([x_cam,y_cam,z_cam])
        r_cam /= np.linalg.norm(r_cam)
        
        R_pitch = np.array([[1,0,0],
                            [0,cos(self.pitch),-sin(self.pitch)],
                            [0,sin(self.pitch),cos(self.pitch)]])
        
        R_yaw = np.array([[cos(self.yaw),-sin(self.yaw),0],
                          [sin(self.yaw),cos(self.yaw),0],
                          [0,0,1]])
        
        ray_world = R_pitch @ R_yaw @ r_cam
        t = -self.altitude/ray_world[2]
        ground_point = ray_world * t
        return ground_point[0],ground_point[1]

        

    def cam_callback(self, IMG):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(IMG, "bgr8") 
            self.detected_mid_point = None
            #frame de yapay zeka obje bulduysa koordinatlarını al ve framede sınırlayıcı kutu çiz
            results = self._infer_model.track(self.frame,True,True)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:  
                    for box in result.boxes:
                        if box.xyxy is not None:
                            
                            #map(int, box.xyxy[0]) komutu ile 2 boyutlu matrisi değişkenlere ayırıyoruz
                            self.x1, self.y1, self.x2, self.y2 = map(int, box.xyxy[0])

                            #objenin orta noktasını buluyoruz
                            self.detected_mid_point = [self.x1 + (abs(self.x2-self.x1)//2),
                                                       self.y1 + (abs(self.y2 - self.y1)//2)]
                            self.get_logger().info(f"detected_mid_point = {self.detected_mid_point}")

                            #sınırlayıcı kutu çiziyoruz
                            cv2.rectangle(self.frame, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 1)
                            if hasattr(result.boxes, 'is_track') and result.boxes.is_track:
                                self.get_logger().info(f"detected ids:{result.boxes.id}")
                                cv2.putText(self.frame,str(int(box.id.item())),(self.x1,self.y1-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                            
                            if self.ID is not None: # takip edilecek obje seçildi ise
                                self.get_logger().info(f"result.names = {box.cls} ID = {self.ID}")
                                if "rover" == str(box.id.names[int(box.cls)]): #seçilen ID ile tespit
                                    continue
                                
                                else:
                                    if self.started == False:

                                        world_x, world_z = self.align_axis()
                                        
                                        self.ukf.x = np.array([world_x, 0, world_z, 0])
                                        self.ukf.P = np.diag([2**2, 1**2, 2**2, 1**2])
                                        Q = np.zeros([4,4])
                                        Q = Q_continuous_white_noise(2, 0.01, 4, 2, True)
                                        R = np.diag([5**2, 5**2])
                                        self.ukf.Q = Q
                                        self.ukf.R = R
                                        
                                        self.ukf.fx = fx
                                        self.ukf.hx = hx
                                        self.start_time = time.time()
                                        self.now_ypr  = time.time()
                                        self.now_move = time.time()
                                        self.started = True
                                    dt = time.time() - self.start_time

                                    self.ukf._dt = dt
                                    self.start_time=time.time()

                                    self.ukf.predict(dt=dt,u=np.array([0,0]))
                                    self.get_logger().info(f"predict x {self.ukf.x_prior} predict.P {self.ukf.P_prior}")
                                    
                                    camera_matrix = np.array([[self.focal_x,0,self.cx],
                                                              [0,self.focal_y,self.cy],
                                                              [0,0,1]])
                                    
                                    # Yaw açısını normalize et (-π, π aralığında)

                                    
                                    R_pitch = np.array([[1,0,0],
                                                        [0,cos(self.pitch),-sin(self.pitch)],
                                                        [0,sin(self.pitch),cos(self.pitch)]])
                
                                    R_yaw = np.array([[cos(self.yaw),-sin(self.yaw),0],
                                                      [sin(self.yaw),cos(self.yaw),0],
                                                      [0,0,1]])
                                    rotation_matrix = R_yaw @ R_pitch
                                    tvec = np.array([0,self.altitude,0])
                                    
                                
                                    
                                    
                            
                                    self.ukf.update(self.detected_mid_point, rotation_matrix=rotation_matrix, translation_vector=tvec, camera_matrix=camera_matrix)
                                    
                                    self.get_logger().info(f"ukf update x = {self.ukf.x_post} ukf update p = {self.ukf.P_post}")
                                    self.get_logger().info(f"align axis {self.align_axis()}")
                                    yaw,pitch = self.calculate_gimbal_angles(self.ukf.x[0], self.ukf.x[2])
                                    self.get_logger().info(f"desired_yaw = {yaw}, desired_pitch = {pitch}")
                                    
                                    predicted_coords = hx(self.ukf.x, rotation_matrix, translation_vector=tvec, camera_matrix=camera_matrix)
                                    self.frame = cv2.circle(self.frame, [int(predicted_coords[0]), int(predicted_coords[1])], 2, (255,0,0), 2)
                                    """
                                    if time.time() - self.now_ypr>7:
                                        
                                        self.gimbal_set_pitch(pitch)
                                        self.gimbal_set_yaw(-yaw * self.p_yaw)
                                        self.now_ypr = time.time()
                                    """
                                    """
                                    if time.time() - self.now_move >15:
                                        
                                        self.move_body_pos(self.ukf.x[2],self.ukf.x[0],0.)
                                        self.now_move = time.time()
                                    """


                                    
                                    

                                    
            
            #ekranın orta noktasını belirten çizgileri çiziyoruz
            cv2.line(self.frame,
                 (FRAME_WIDTH // 2, (FRAME_HEIGHT // 2) + 3),
                 (FRAME_WIDTH // 2, (FRAME_HEIGHT // 2) - 3),
                 (0, 255, 0), 1)

            cv2.line(self.frame,
                (FRAME_WIDTH // 2 + 3, FRAME_HEIGHT // 2),
                (FRAME_WIDTH // 2 - 3, FRAME_HEIGHT // 2),
                (0, 255, 0), 1)

            #ekranın orta noktasından objenin orta noktasına çizgi çiziyoruz
            if self.detected_mid_point is not None:
                cv2.line(self.frame, tuple(self.detected_mid_point), tuple(FRAME_MIDDLE), (0,255,0), 1)
            
            #görüntüyü ekrana yazdırıyoruz
            cv2.imshow("camera", self.frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  
                cv2.destroyAllWindows()

        except Exception as e:
            self.get_logger().error(f"Error in cam_callback: {str(e)}")

    #gimbal kontrol komutları, gz sim topiclerinden ros2 topiclerine köprü kurmaktansa 
    #terminal komutlarını kullanmak için subprocess kütüphanesini kullanıyoruz
    def gimbal_set_yaw(self,yaw):
        self.yaw=yaw
        commands=[
            "gz","topic",
            "-t","/gimbal/cmd_yaw",
            "-m","gz.msgs.Double",
            "-p",f'data:{yaw}'
        ]
        subprocess.Popen(commands)
        self.get_logger().info(f"yaw = {yaw}")

    def gimbal_set_roll(self,roll):
        commands=[
            "gz","topic",
            "-t","/gimbal/cmd_roll",
            "-m","gz.msgs.Double",
            "-p",f'data:{roll}'
        ]
        subprocess.Popen(commands)
        self.roll=roll

    def gimbal_set_pitch(self,pitch):
        
        commands=[
            "gz","topic",
            "-t","/gimbal/cmd_pitch",
            "-m","gz.msgs.Double",
            "-p",f'data:{pitch}'
        ]
        subprocess.Popen(commands)
        self.pitch=pitch

    #yolo modelimizi eğitebilmek için fotoğraf çekme servisimiz
    def take_picture(self,request,response):
        picture_directory = r"/home/baran/ardu_ws/src/main/YOLO_images"
        os.chdir(picture_directory)
        filename = str(time.time()) + ".jpeg"
        cv2.imwrite(filename,self.frame)
        return response
    
    def arm_motors(self,arm:bool):
        self.req_arm.arm = arm
        self._client_arm.call(self.req_arm)
    
    def takeoff(self,altitude:float):
        self.req_takeoff.alt = altitude
        self._client_takeoff.call(self.req_takeoff)

    def mode_switch(self,mode:int):
        self.req_mode.mode=mode
        self._client_mode.call(self.req_mode)
    
    #ned(North-East-Down) koordinatlarına göre hareket komutu
    def move_ned_velocity(self,x,y,z,yaw,duration):
        self.req_ned_vel.x = x
        self.req_ned_vel.y = y
        self.req_ned_vel.z = z
        self.req_ned_vel.yaw = yaw
        self.req_ned_vel.sec = duration
        self._client_move_ned_vel.call_async(self.req_ned_vel)
    
    #body(dronun baktığı yön x düzlemi olacak şekilde) koordinatlarına göre hareket komutu
    def move_body_velocity(self,x,y,z,yaw,duration):
        rotation = np.array([[cos(self.drone_yaw), -sin(self.drone_yaw)],
                                [sin(self.drone_yaw),cos(self.drone_yaw)]])
        state = np.array([x,y])
        new_x,new_y = rotation @ state
        
        self.req_body_vel.x = new_x
        self.req_body_vel.y = new_y
        self.req_body_vel.z = z
        self.req_body_vel.yaw = yaw  # yaw parametresi eklendi
        self.req_body_vel.sec = duration
        future = self._client_move_body_vel.call_async(self.req_body_vel)
    
    def move_body_pos (self,x,y,z):
        self.req_body_pos.x = x
        self.req_body_pos.y = y
        self.req_body_pos.z = z
        self._client_move_body_pos.call_async(self.req_body_pos)
    
    def move_ned_pos (self,x,y,z):
        self.req_ned_pos.x = x
        self.req_ned_pos.y = y
        self.req_ned_pos.z = z
        self._client_move_ned_pos.call_async(self.req_ned_pos)


def main():
    rclpy.init()
    control_node=Control(1.35,1.,0.1,0.01)
    control_node.gimbal_set_pitch(0.65)
    control_node.gimbal_set_yaw(0.0)
    rclpy.spin(control_node)
    """
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(control_node)
    executor.spin()
    """
if __name__ == '__main__':
    main()