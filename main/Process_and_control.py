"""
CyberFlight takimi görüntü işleme ve görev kodu. 
bu dosyada çalisma hizini arttirmak icin python listeleri yerine numpy arraylari kullanilmistir.
Ayrica bu kodda geçen x,y,z koordinat sistemi dronun gimbalina göredir yani y ekseni gimbalin ileri gerisi, x sagi solu ve z irtifasidir. 
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
from main.Kalman_filter import UKF,Q_continuous



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
        self.get_logger().info(f"focal_x {self.focal_x} focal_y { self.focal_y}")
        
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        #numpy arraylerde append fonksiyonu python listelerindeki gibi olmadığından normal liste
        self.signal_queue =[[0,0],[0,0]]

        #ROS'tan opencv ye çevirmek için köprü
        self.bridge=CvBridge()

        self._infer_model = YOLO("/home/baran/ardu_ws/src/main/inference_models/best.pt")
        self.detected_mid_point = None
        self.ID = None
        self.started = False
    #clientlar
        #Body
        self._client_move_body = self.create_client(Waypoint,"mavlink_copter/move_velocity_body",)
        while not self._client_move_body.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('servis move_body servisi bekleniyor...')
        self.req_body = Waypoint.Request()
        
        #NED
        self._client_move_ned = self.create_client(Waypoint,"mavlink_copter/move_velocity_ned",)
        while not self._client_move_ned.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('servis move_ned servisi bekleniyor...')
        self.req_ned = Waypoint.Request()
        
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
    
    def assign_altitude(self,msg): self.altitude = msg.data 
    def assign_yaw(self,msg): self.drone_yaw = msg.data
    
    def fx(self,X,dt,u):
        # State: [x, vx, y, vy] - position and velocity in drone body frame
        new_x = X[0] + X[1] * dt
        new_vel_x = X[1]  # Constant velocity model
        new_y = X[2] + X[3] * dt
        new_vel_y = X[3]  # Constant velocity model
        return np.array([new_x, new_vel_x, new_y, new_vel_y])

    def hx(self, X, focal_x, focal_y, frame_middle, altitude, pitch, yaw):
        """
        Measurement model: projects 3D position to 2D image coordinates
        X: [x, vx, y, vy] - position in drone body frame
        Returns: (u, v) image coordinates
        """
        # Extract position from state
        x_pos = X[0]  # Right-left position in drone body frame
        y_pos = X[2]  # Forward-backward position in drone body frame
        
        # Create 3D point (matching original working version)
        object_point = np.array([[x_pos, y_pos, 0]], dtype=np.float32)
        
        # Camera intrinsic matrix
        camera_matrix = np.array([
            [focal_x, 0, frame_middle[0]],
            [0, focal_y, frame_middle[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Rotation matrices for gimbal orientation
        R_yaw = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        R_pitch = np.array([
            [1,0 ,0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0,math.sin(pitch),math.cos(pitch)]
        ], dtype=np.float32)
        
        # Combined rotation
        R_cam = R_yaw @ R_pitch
        rvec,_ = cv2.Rodrigues(R_cam)
        tvec = np.array((0., 0.,altitude), dtype=np.float32)
            
        image_points,_ = cv2.projectPoints(object_point, rvec, tvec, camera_matrix, dist_coeffs)
        u, v = image_points[0][0]
        
        return u, v

    def debug_coordinate_transformations(self):
        """
        Debug method to test coordinate transformations
        """
        # Test with known ground truth
        test_x, test_y = 5.0, 5.0  # Expected position in drone body frame
        test_state = np.array([test_x, 0.0, test_y, 0.0])
        
        # Test hx function
        image_coords = self.hx(test_state, self.focal_x, self.focal_y, FRAME_MIDDLE, self.altitude, self.pitch, self.yaw)
        self.get_logger().info(f"Test state {test_state} -> Image coords {image_coords}")
        
        # Test reverse transformation (what align_axis would give)
        # Simulate detected point at the image coordinates
        self.detected_mid_point = [int(image_coords[0]), int(image_coords[1])]
        body_x, body_y, _, _ = self.align_axis()
        self.get_logger().info(f"Image coords {self.detected_mid_point} -> Body coords ({body_x:.2f}, {body_y:.2f})")
        
        # Check if they match
        error_x = abs(test_x - body_x)
        error_y = abs(test_y - body_y)
        self.get_logger().info(f"Coordinate transformation error: x={error_x:.3f}, y={error_y:.3f}")
        
        # Test with different positions
        test_positions = [(0, 5), (5, 0), (-3, 4), (2, -1)]
        for tx, ty in test_positions:
            test_state = np.array([tx, 0.0, ty, 0.0])
            image_coords = self.hx(test_state, self.focal_x, self.focal_y, FRAME_MIDDLE, self.altitude, self.pitch, self.yaw)
            self.detected_mid_point = [int(image_coords[0]), int(image_coords[1])]
            body_x, body_y, _, _ = self.align_axis()
            error_x = abs(tx - body_x)
            error_y = abs(ty - body_y)
            self.get_logger().info(f"Position ({tx}, {ty}) -> Image {image_coords} -> Body ({body_x:.2f}, {body_y:.2f}) -> Error ({error_x:.3f}, {error_y:.3f})")
        
        return error_x < 0.1 and error_y < 0.1  # Return True if transformation is accurate

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
        """
        Calculate gimbal angles to point at target
        target_x, target_y: target position in drone body frame
        Returns: (yaw, pitch) angles for gimbal
        """
        # Calculate distance and angles in body frame
        distance = math.sqrt(target_x**2 + target_y**2)
        
        if distance < 1e-6:
            return 0.0, 0.0
        
        # Calculate angles to point at target
        yaw = math.atan2(target_x, target_y)  # Right-left angle
        pitch = math.atan2(-self.altitude, distance)  # Up-down angle
        
        return yaw, pitch

    def align_axis(self):
        """
        Calculates the ground intersection point of the pixel using camera intrinsics and extrinsics.
        Returns: (x, y, desired_yaw, desired_pitch) in drone body frame
        """
        if self.detected_mid_point is None:
            self.get_logger().warn("Nesne tespit edilmedi, hizalama yapilamiyor.")
            return 0.0, 0.0, self.yaw, self.pitch

        detect_x, detect_y = self.detected_mid_point
        cx = FRAME_WIDTH / 2
        cy = FRAME_HEIGHT / 2

        # HFOV'dan fx hesapla
        fx = (FRAME_WIDTH / 2) / math.tan(self.HFOV / 2)
        fy = (FRAME_HEIGHT/2)/math.tan(self.VFOV/2)  # Kamera kare kabul edildi

        # Piksel koordinatını normalize et
        x_cam = (detect_x - cx) / fx
        y_cam = (detect_y - cy) / fy
        z_cam = 1.0  # Görüntü düzleminde ileri yön

        # Görüntü düzleminden gelen vektörü birim vektöre çevir
        ray_camera = np.array([x_cam, y_cam, z_cam])
        ray_camera /= np.linalg.norm(ray_camera)

        # Kameranın pitch ve yaw açılarına göre dönüş matrisi
        R_pitch = np.array([
            [1, 0, 0],
            [0, math.cos(self.pitch), -math.sin(self.pitch)],
            [0, math.sin(self.pitch), math.cos(self.pitch)]
        ])

        R_yaw = np.array([
            [math.cos(self.yaw), -math.sin(self.yaw), 0],
            [math.sin(self.yaw), math.cos(self.yaw), 0],
            [0, 0, 1]
        ])

        # Kamera ışınını dünya eksenine çevir
        ray_world = R_yaw @ R_pitch @ ray_camera


        t = -self.altitude / ray_world[2]  # z eksenini sıfırlamak için ölçek
        ground_point = t * ray_world

        body_x = ground_point[0]
        body_y = ground_point[1]

        desired_yaw = math.atan2(body_x, body_y)
        desired_pitch = math.atan2(-ground_point[2], np.linalg.norm(ground_point[:2]))

        self.get_logger().info(
            f"[align_axis] Ground XY: ({body_x:.2f}, {body_y:.2f}), "
            f"desired_yaw: {math.degrees(desired_yaw):.2f}°, desired_pitch: {math.degrees(desired_pitch):.2f}°, "
            f"altitude: {self.altitude:.2f}m"
        )

        return body_x, body_y, desired_yaw, desired_pitch

        

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
                                if self.started == False: # sadece bir kez başlangıçta çalışacak
                                    Q=np.zeros([4,4])
                                    Q[0:2,0:2] = Q_continuous(2,1,0.001)
                                    Q[2:4,2:4] = Q_continuous(2,1,0.001)
                                    self.get_logger().info(f"içerideyiz")
                                    P=np.diag([5**2,2**2,5**2,2**2])
                                    R = np.diag([5.,5.]) #ölçüm parazit matrisi
                                    x,y,_,_= self.align_axis()  # Use align_axis for proper initialization
                                    self.get_logger().info(f"align axis feedback x = {x} y = {y}")
                                    self.ukf = UKF(
                                        dim_x=4,
                                        dim_z=2,
                                        fx=self.fx,
                                        hx=self.hx,
                                        x=np.array([x,0,y,0]),
                                        P=P,
                                        R=R,
                                        Q=Q,
                                        residual_fnc=np.subtract,
                                        dt=0.001
                                    )
                                    self.Wm, self.Wc, self.lambda_ = self.ukf.Van_der_merve_weights(0.3,2,0)
                                    
                                    self.start_time = time.time()
                                    self.started = True
                                    self.last_cmd = time.time()
                                dt = time.time() - self.start_time
                                self.ukf.dt = dt

                                
                                self.start_time = time.time()
                                self.get_logger().info(f"dt = {dt}")
                                sigmas = self.ukf.sigma_points(self.ukf.x,self.ukf.P,self.lambda_)
                                self.get_logger().info(f"sigmas = {sigmas}")
                                
                                self.ukf.predict(sigmas,dt,np.array([0,0,0]))
                                self.get_logger().info(f"self.ukf predict x = {self.ukf.x}")
                                self.get_logger().info(f"self.ukf predict P = {self.ukf.P}")

                                self.ukf.update(self.detected_mid_point,self.focal_x,self.focal_y,FRAME_MIDDLE,self.altitude,self.pitch,self.yaw)
                                self.get_logger().info(f"self.ukf update x = {self.ukf.x}")
                                self.get_logger().info(f"self.ukf update P = {self.ukf.P}")
                                
                                #filtre tahmini noktası
                                object_coords = self.hx(self.ukf.x,self.focal_x,self.focal_y,FRAME_MIDDLE,self.altitude,self.pitch,self.yaw)
                                self.frame = cv2.circle(self.frame, (int(object_coords[0]), int(object_coords[1])), 5, (0, 0, 255), -1)
                                
                                #align_axis tahmini noktası
                                align_axis_coords_x,align_axis_coords_y,pitch,yaw = self.align_axis()
                                align_axis_coords_hx = self.hx(np.array([align_axis_coords_x,0,align_axis_coords_y,0]),self.focal_x,self.focal_y,FRAME_MIDDLE,self.altitude,self.pitch,self.yaw)
                                self.frame = cv2.circle(self.frame, (int(align_axis_coords_hx[0]), int(align_axis_coords_hx[1])), 5, (255, 0, 0), -1)
                                
                                #gerçek değer
                                true_value=self.hx(np.array([5,0,5,0]),self.focal_x,self.focal_y,FRAME_MIDDLE,self.altitude,self.pitch,self.yaw)
                                self.frame=cv2.circle(self.frame,(int(true_value[0]),int(true_value[1])),5,(0,255,0),2)
                                self.get_logger().info(f"true_value = {true_value} align_axis_coords_hx = {align_axis_coords_hx} object_coords = {object_coords}")
                                self.get_logger().info(f"current pitch = {self.pitch} current yaw = {self.yaw}")
                                
                                # Gimbal control - use absolute angles
                                now = time.time()
                                if now - self.last_cmd > 10:
                                    self.last_cmd = now
                                    
                                    target_x = self.ukf.x[0]
                                    target_y = self.ukf.x[2]
                                    
                                    # Calculate desired gimbal angles to point at target
                                    desired_yaw, desired_pitch = self.calculate_gimbal_angles(target_x, target_y)
                                    
                                    # Set absolute gimbal angles (not relative)
                                    self.gimbal_set_pitch(-desired_pitch)
                                    
                                    self.get_logger().info(f"Target: ({target_x:.2f}, {target_y:.2f}) -> Gimbal: yaw={desired_yaw:.3f}, pitch={desired_pitch:.3f}")
                                
                        else:
                            self.detected_mid_point = None
            
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
    def move_ned(self,x,y,z,duration):
        self.req_ned.x = x
        self.req_ned.y = y
        self.req_ned.z = z
        self.req_ned.sec = duration
        future = self._client_move_ned.call_async(self.req_ned)
        rclpy.spin_until_future_complete(self, future)
    
    #body(dronun baktığı yön x düzlemi olacak şekilde) koordinatlarına göre hareket komutu
    def move_body(self,x,y,z,yaw,duration):
        self.req_body.lat = x
        self.req_body.lon = y
        self.req_body.alt = z
        self.req_body.x = 1.2
        self.req_body.y = 1.2
        self.req_body.z = 0.0
        self.req_body.yaw=yaw
        self.req_body.sec = duration
        future = self._client_move_body.call_async(self.req_body)
        rclpy.spin_until_future_complete(self, future)

def main():
    rclpy.init()
    control_node=Control(1.,1.,0.1,0.01)
    control_node.gimbal_set_pitch(0.9)
    control_node.gimbal_set_yaw(0.2)
    control_node.align_axis()
    rclpy.spin(control_node)
    """
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(control_node)
    executor.spin()
    """
if __name__ == '__main__':
    main()