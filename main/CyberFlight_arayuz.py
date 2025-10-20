import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
import cv2
import sys
import serial.tools.list_ports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QLineEdit, QTextEdit, QVBoxLayout, QGridLayout, QGroupBox,
    QMessageBox, QMenuBar, QStatusBar, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QAction, QPalette, QColor, QIcon, QPixmap, QPainter
from main_msgs.srv import Arm,Takeoff
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import QFileDialog

class DroneSignals(QObject):
    connection_changed = pyqtSignal(bool)
    armed_changed = pyqtSignal(bool)
    telemetry_updated = pyqtSignal(dict)
    message_received = pyqtSignal(str)
    
class VideoPlaybackThread(QThread):
    frame_received = pyqtSignal(QImage)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qt_img = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
            self.frame_received.emit(qt_img)

            QThread.msleep(33)  # ~30 FPS

        cap.release()

    def stop(self):
        self.running = False






class DroneInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.signals = DroneSignals()

        # Uçuş modu değişkeni burada tanımlanmalı
        self.current_flight_mode = "GUIDED"

        self.setup_ui() # <--- Artık bu fonksiyon çağrıldığında değişken mevcut olacak
        self.connect_signals()
        
        self.log_message(">>> Sistem başlatıldı <<<")
        self.log_message(">>> Bağlantı için COM port ve baudrate seçin <<<")

        # ROS kamera thread
        #self.ros_cam_thread = RosCameraThread()
        #self.ros_cam_thread.frame_received.connect(self.update_camera_display)
        #self.ros_cam_thread.yaw_alt_updated.connect(self.update_yaw_altitude)
        #self.ros_cam_thread.arm_service_result.connect(self.on_arm_result)
        #self.ros_cam_thread.takeoff_service_result.connect(self.on_takeoff_result)
        #self.ros_cam_thread.start()

    def setup_ui(self):
        self.setWindowTitle("Drone Kontrol Arayüzü")
        self.setMinimumSize(1024, 768)
        self.showMaximized()

        self.selected_com = None
        self.selected_baud = "9600"
        self.armed = False
        self.connected = False

        self.setup_colors()

        central = QWidget(self)
        central.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCentralWidget(central)
        
        self.main_layout = QGridLayout(central)
        self.main_layout.setContentsMargins(15, 10, 15, 5)
        self.main_layout.setHorizontalSpacing(10)
        self.main_layout.setVerticalSpacing(10)

       # Bileşenleri oluştur
        self.create_connection_panel()
        self.create_menu_bar()
        self.create_camera_display()
        self.create_indicator_panels()
        self.gimbal_group = self.create_gimbal_panel()
        self.movable_group = self.create_movable_panel()
        self.control_frame = self.create_control_panel()
        self.create_console()
        self.create_status_bar()

        # YENİ: Uçuş modu seçim paneli
        self.flight_mode_group = self.create_flight_mode_panel()

        # Grid düzeni - YENİ DÜZEN (gimbal ve movable arasına flight_mode_group eklendi)
        self.main_layout.addWidget(self.camera_frame, 0, 0, 3, 2)
        self.main_layout.addWidget(self.connection_frame, 0, 2, 1, 2)
        self.main_layout.addWidget(self.heading_indicator, 1, 2)
        self.main_layout.addWidget(self.attitude_indicator, 1, 3)
        self.main_layout.addWidget(self.control_frame, 2, 2, 1, 2)
        self.main_layout.addWidget(self.gimbal_group, 3, 0, 1, 1)  # 2 sütundan 1'e değişti
        self.main_layout.addWidget(self.flight_mode_group, 3, 1, 1, 1)  # YENİ EKLENDİ
        self.main_layout.addWidget(self.movable_group, 3, 2, 1, 2)   # 2 sütundan 2'ye değişti
        self.main_layout.addWidget(self.log_console, 4, 0, 1, 4)

        # Grid esneklik ayarları
        self.main_layout.setColumnStretch(0, 2)  # Gimbal paneli için
        self.main_layout.setColumnStretch(1, 1)  # Yeni uçuş modu paneli için
        self.main_layout.setColumnStretch(2, 1)  # Hedef takip paneli için
        self.main_layout.setColumnStretch(3, 1)  # Hedef takip paneli için
        self.main_layout.setRowStretch(0, 1)     # Üst satırlar
        self.main_layout.setRowStretch(1, 1)     # Orta satırlar
        self.main_layout.setRowStretch(2, 1)     # Alt satırlar
        self.main_layout.setRowStretch(3, 1)     # Yeni eklenen paneller
        self.main_layout.setRowStretch(4, 1)     # Konsol (daha az yer kaplasın)
    def play_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Video Seç", "", "Video Dosyaları (*.mp4 *.avi *.mov)")
        if not video_path:
            return

        self.log_message(f"🎞️ Video başlatılıyor: {video_path}")

        # ROS thread'i durdur (görsel çakışmayı engellemek için)
        #self.ros_cam_thread.stop()
        #self.ros_cam_thread.wait()

        # Video oynatma thread'i başlat
        self.video_thread = VideoPlaybackThread(video_path)
        self.video_thread.frame_received.connect(self.update_camera_display)
        self.video_thread.start()
        
    def create_flight_mode_panel(self):
        flight_group = QGroupBox("UÇUŞ MODU SEÇİMİ")
        self.apply_groupbox_style(flight_group)
        layout = QVBoxLayout(flight_group)
        layout.setSpacing(10)
        
        # Uçuş modları için butonlar
        self.guided_btn = self.create_flight_mode_button("GUIDED", self.set_guided_mode)
        self.loiter_btn = self.create_flight_mode_button("LOITER", self.set_loiter_mode)
        self.land_btn = self.create_flight_mode_button("LAND", self.set_land_mode)
        self.rtl_btn = self.create_flight_mode_button("RTL", self.set_rtl_mode)
        
        layout.addWidget(self.guided_btn)
        layout.addWidget(self.loiter_btn)
        layout.addWidget(self.land_btn)
        layout.addWidget(self.rtl_btn)
        
        # Seçilen modu gösteren etiket
        self.flight_mode_label = QLabel(f"Seçilen Mod: {self.current_flight_mode}")
        self.flight_mode_label.setStyleSheet(f"""
            font: bold 14px;
            color: {self.colors['green_primary'].name()};
            background-color: {self.colors['dark_secondary'].name()};
            border-radius: 8px;
            padding: 8px;
            text-align: center;
        """)
        layout.addWidget(self.flight_mode_label)
        
        return flight_group
    

    def create_flight_mode_button(self, text, callback):
        btn = QPushButton(text)
        btn.setFixedHeight(40)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['dark_secondary'].name()};
                color: {self.colors['text_color'].name()};
                border: 2px solid {self.colors['blue_primary'].name()};
                border-radius: 8px;
                font: bold 12px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['blue_primary'].name()};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['dark_primary'].name()};
            }}
        """)
        btn.clicked.connect(callback)
        return btn
    
    def set_guided_mode(self):
        self.set_flight_mode("GUIDED")
        
    def set_loiter_mode(self):
        self.set_flight_mode("LOITER")
        
    def set_land_mode(self):
        self.set_flight_mode("LAND")
        
    def set_rtl_mode(self):
        self.set_flight_mode("RTL")
        
    def set_flight_mode(self, mode):
        self.current_flight_mode = mode
        self.flight_mode_label.setText(f"Seçilen Mod: {mode}")
        self.log_message(f"Uçuş modu değiştirildi: {mode}")
        self.status_bar.showMessage(f"Geçerli uçuş modu: {mode}", 5000)
        
        # Buton vurgulama (seçilen modu belirginleştir)
        all_btns = [self.guided_btn, self.loiter_btn, self.land_btn, self.rtl_btn]
        for btn in all_btns:
            if btn.text() == mode:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.colors['green_primary'].name()};
                        color: {self.colors['dark_primary'].name()};
                        border: 2px solid {self.colors['dark_primary'].name()};
                        border-radius: 8px;
                        font: bold 14px;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {self.colors['dark_secondary'].name()};
                        color: {self.colors['text_color'].name()};
                        border: 2px solid {self.colors['blue_primary'].name()};
                        border-radius: 8px;
                        font: bold 12px;
                    }}
                """)



    def create_gimbal_panel(self):
        gimbal_group = QGroupBox("GIMBAL KONTROL")
        self.apply_groupbox_style(gimbal_group)
        layout = QGridLayout(gimbal_group)
        
        # Yaw kontrolü
        layout.addWidget(QLabel("YAW:"), 0, 0)
        self.yaw_value = QLabel("1.3 rad")
        layout.addWidget(self.yaw_value, 0, 1)
        
        # Pitch kontrolü
        layout.addWidget(QLabel("PITCH:"), 1, 0)
        self.pitch_value = QLabel("0.22 rad")
        layout.addWidget(self.pitch_value, 1, 1)
        
        # Roll kontrolü
        layout.addWidget(QLabel("ROLL:"), 2, 0)
        self.roll_value = QLabel("0.0 rad")
        layout.addWidget(self.roll_value, 2, 1)
        
        return gimbal_group
    
    def create_movable_panel(self):
        movable_group = QGroupBox("HEDEF TAKİP")
        self.apply_groupbox_style(movable_group)
        layout = QGridLayout(movable_group)
        
        # ID seçimi
        layout.addWidget(QLabel("Hedef ID:"), 0, 0)
        self.target_id = QLineEdit()
        self.target_id.setPlaceholderText("ID girin")
        layout.addWidget(self.target_id, 0, 1)
        
        # Takip butonu
        self.track_btn = QPushButton("Takip Et")
        layout.addWidget(self.track_btn, 0, 2)
        
        # Konum bilgileri
        layout.addWidget(QLabel("x:"), 1, 0)
        self.pos_x = QLabel("5m")
        layout.addWidget(self.pos_x, 1, 1)
        
        layout.addWidget(QLabel("y:"), 2, 0)
        self.pos_y = QLabel("5m")
        layout.addWidget(self.pos_y, 2, 1)
        
        # Hız bilgileri
        layout.addWidget(QLabel("Hız x:"), 3, 0)
        self.vel_x = QLabel("0.5 m/s")
        layout.addWidget(self.vel_x, 3, 1)
        
        layout.addWidget(QLabel("Hız y:"), 4, 0)
        self.vel_y = QLabel("0.5 m/s")
        layout.addWidget(self.vel_y, 4, 1)
        
        return movable_group


    def arm_drone(self):
        self.armed = not self.armed
        self.log_message(f"ARM komutu gönderildi → {self.armed}")
        #self.ros_cam_thread.call_arm_service(self.armed)

    def on_arm_result(self, success):
        if success:
            self.log_message("✅ ARM servisi başarılı")
            self.signals.armed_changed.emit(self.armed)
        else:
            self.log_message("❌ ARM servisi başarısız oldu!")
            self.armed = False
            self.signals.armed_changed.emit(self.armed)

    def takeoff_drone(self):
        """Kalkış komutunu gönder (güncellenmiş versiyon)"""
        try:
            hedef_alt = float(self.altitude_input.text())
            if hedef_alt < 0.5 or hedef_alt > 20.0:
                raise ValueError("İrtifa 0.5-20.0m aralığında olmalı")
                
            self.log_message(f"Kalkış komutu gönderildi, hedef: {hedef_alt:.1f} m")
            #self.ros_cam_thread.call_takeoff_service(hedef_alt)
            
        except ValueError as e:
            self.log_message(f"❌ Geçersiz irtifa değeri! {str(e)}")
            QMessageBox.warning(
                self,
                "Geçersiz Değer",
                f"Lütfen 0.5 ile 20.0 metre arasında bir değer girin\n\nHata: {str(e)}"
            )
            self.altitude_input.setFocus()

    def on_takeoff_result(self, success):
        if success:
            self.log_message("✅ Kalkış başarılı!")
        else:
            self.log_message("❌ Kalkış servisi başarısız!")
    def update_yaw_altitude(self, yaw, altitude):
        self.telemetry_labels['altitude'].setText(f"📏 İRTİFA: {altitude:.1f} m")
        self.heading_dial.setText(f"↻\n{yaw:.0f}°")
    
    def paintEvent(self, event):
        """Arka plan resmini çizmek için paint event'i override et"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        logo = QPixmap("logo.png")
        
        if not logo.isNull():
            window_size = self.size()
            logo_width = int(window_size.width() * 0.4)
            logo_height = int(window_size.height() * 0.4)
            logo_size = logo.scaled(
                logo_width, 
                logo_height, 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            x = (window_size.width() - logo_size.width()) / 2
            y = (window_size.height() - logo_size.height()) / 2
            
            painter.setOpacity(0.2)
            painter.drawPixmap(int(x), int(y), logo_size)
        
        painter.end()

    def setup_colors(self):
        """Renk paletini ayarla"""
        self.colors = {
            'dark_primary': QColor(44, 62, 80),
            'dark_secondary': QColor(52, 73, 94),
            'blue_primary': QColor(52, 152, 219),
            'red_primary': QColor(231, 76, 60),
            'green_primary': QColor(46, 204, 113),
            'text_color': QColor(236, 240, 241),
            'text_secondary': QColor(149, 165, 166)
        }

    def connect_signals(self):
        """Sinyal-slot bağlantılarını yap"""
        self.signals.connection_changed.connect(self.update_connection_status)
        self.signals.armed_changed.connect(self.update_armed_status)
        self.signals.telemetry_updated.connect(self.update_telemetry)
        self.signals.message_received.connect(self.log_message)

    def create_menu_bar(self):
        """Menü çubuğunu oluştur"""
        menubar = self.menuBar()
        self.apply_menu_styles(menubar)

        # Dosya menüsü
        file_menu = menubar.addMenu("Dosya")
        file_menu.addAction(self.create_action("Aç", "Ctrl+O", self.open_file))
        file_menu.addAction(self.create_action("Kaydet", "Ctrl+S", self.save_file))
        file_menu.addAction(self.create_action("Video Aç", "", self.play_video))
        file_menu.addSeparator()
        file_menu.addAction(self.create_action("Çıkış", "Ctrl+Q", self.exit_app))

        # Bağlantı menüsü
        connection_menu = menubar.addMenu("Bağlantı")
        self.create_com_port_menu(connection_menu)
        self.create_baudrate_menu(connection_menu)
        connection_menu.addAction(self.create_action("Bağlan", "Ctrl+B", self.connect_drone))

        # Yardım menüsü
        help_menu = menubar.addMenu("Yardım")
        help_menu.addAction(self.create_action("Hakkında", "", self.show_about))

    def apply_menu_styles(self, menubar):
        """Menü stillerini uygula"""
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: {self.colors['dark_secondary'].name()};
                color: {self.colors['text_color'].name()};
                padding: 5px;
            }}
            QMenuBar::item {{
                padding: 5px 10px;
                background: transparent;
                border-radius: 4px;
            }}
            QMenuBar::item:selected {{
                background: {self.colors['dark_primary'].name()};
            }}
            QMenu {{
                background-color: {self.colors['dark_secondary'].name()};
                border: 1px solid {self.colors['dark_primary'].name()};
                margin: 2px;
            }}
            QMenu::item:selected {{
                background-color: {self.colors['blue_primary'].name()};
            }}
        """)

    def create_action(self, text, shortcut, callback):
        """Yeni bir menü aksiyonu oluştur"""
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        return action

    def create_com_port_menu(self, parent_menu):
        """Dinamik COM Port menüsü oluştur"""
        self.com_menu = parent_menu.addMenu("COM Port")
        self.com_menu.aboutToShow.connect(self.update_com_ports)
        self.com_actions = []
        self.update_com_ports()

    def update_com_ports(self):
        """COM portlarını güncelle"""
        self.com_menu.clear()
        self.com_actions = []
        
        try:
            ports = serial.tools.list_ports.comports(include_links=True)
            if not ports:
                self.log_message("Hiç COM portu algılanamadı.")  # Hatalı print kaldırıldı
                action = self.create_action("Bağlı cihaz yok", "", lambda: None)
                action.setEnabled(False)
                self.com_menu.addAction(action)
                return
                
            for port in ports:
                description = port.description if port.description else "Cihaz"
                display_text = f"{port.device} ({description})"
                action = self.create_action(display_text, "", lambda _, p=port.device: self.set_com_port(p))
                action.setCheckable(True)
                if self.selected_com == port.device:
                    action.setChecked(True)
                self.com_actions.append(action)
                self.com_menu.addAction(action)
        except Exception as e:
            self.log_message(f"Port tarama hatası: {str(e)}")

    def create_baudrate_menu(self, parent_menu):
        """Baudrate menüsünü oluştur"""
        baud_menu = parent_menu.addMenu("Baudrate")
        self.baud_actions = []
        for rate in ["9600", "19200", "57600", "115200"]:
            action = self.create_action(rate, "", lambda r=rate: self.set_baudrate(r))
            action.setCheckable(True)
            self.baud_actions.append(action)
            baud_menu.addAction(action)
        self.baud_actions[0].setChecked(True)

    def create_camera_display(self):
        """Kamera görüntü alanını oluştur"""
        self.camera_frame = QGroupBox("CANLI KAMERA GÖRÜNTÜSÜ")
        self.apply_groupbox_style(self.camera_frame)
        
        camera_layout = QVBoxLayout(self.camera_frame)
        camera_layout.setSpacing(10)

        self.camera_display = QLabel()
        self.camera_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_display.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {self.colors['dark_primary'].name()}, stop:1 {self.colors['blue_primary'].name()});
            border-radius: 15px;
            border: 2px solid {self.colors['dark_secondary'].name()};
        """)
        self.camera_display.setMinimumSize(640, 480)
        camera_layout.addWidget(self.camera_display)

        self.camera_status = QLabel("⏸️ KAMERA BAĞLANTISI YOK")
        self.camera_status.setStyleSheet(f"""
            color: {self.colors['text_secondary'].name()};
            font-style: italic;
            padding: 5px;
        """)
        camera_layout.addWidget(self.camera_status)

        self.main_layout.addWidget(self.camera_frame, 0, 0, 3, 2)

    def update_camera_display(self, qt_img: QImage):
        pix = QPixmap.fromImage(qt_img).scaled(
            self.camera_display.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_display.setPixmap(pix)
        self.camera_status.setText("✅ KAMERA AKIŞI") 

    

    def create_connection_panel(self):
        self.connection_frame = QGroupBox("BAĞLANTI VE TELEMETRİ")  # self. ekleyerek sınıf değişkeni yap
        self.apply_groupbox_style(self.connection_frame)
        
        connection_layout = QGridLayout(self.connection_frame)
        connection_layout.setSpacing(10)

        self.connection_status = QLabel("🔴 BAĞLI DEĞİL")
        self.connection_status.setStyleSheet(f"""
            font: bold 16px; 
            color: {self.colors['red_primary'].name()};
        """)
        connection_layout.addWidget(self.connection_status, 0, 0, 1, 2)

        self.telemetry_labels = {
            'altitude': self.create_telemetry_label("İRTİFA: -- m", "📏"),
            'velocity': self.create_telemetry_label("HIZ: -- m/s", "🚀"),
            'battery': self.create_telemetry_label("PİL: -- %", "🔋"),
            'gps': self.create_telemetry_label("UYDU: --", "🛰️")
        }

        connection_layout.addWidget(self.telemetry_labels['altitude'], 1, 0)
        connection_layout.addWidget(self.telemetry_labels['velocity'], 1, 1)
        connection_layout.addWidget(self.telemetry_labels['battery'], 2, 0)
        connection_layout.addWidget(self.telemetry_labels['gps'], 2, 1)

    def create_telemetry_label(self, text, icon):
        """Telemetri etiketi oluştur"""
        label = QLabel(f"{icon} {text}")
        label.setStyleSheet(f"""
            font: bold 14px;
            color: {self.colors['text_color'].name()};
            padding: 8px;
            background-color: {self.colors['dark_secondary'].name()};
            border-radius: 8px;
        """)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumWidth(150)
        return label

    def create_indicator_panels(self):
        self.heading_indicator = QGroupBox("YÖN GÖSTERGESİ")
        self.apply_groupbox_style(self.heading_indicator)
        
        heading_layout = QVBoxLayout(self.heading_indicator)
        self.heading_dial = self.create_dial_indicator("N\n360°", self.colors['blue_primary'])
        heading_layout.addWidget(self.heading_dial)

        self.attitude_indicator = QGroupBox("EĞİM GÖSTERGESİ")
        self.apply_groupbox_style(self.attitude_indicator)
        
        attitude_layout = QVBoxLayout(self.attitude_indicator)
        self.attitude_dial = self.create_dial_indicator("⬆\n0°", self.colors['green_primary'])
        attitude_layout.addWidget(self.attitude_dial)

        self.main_layout.addWidget(self.heading_indicator, 1, 2)  # Sağ üst (0,3)
        self.main_layout.addWidget(self.attitude_indicator, 1, 3)  # Altında (1,3)



    def create_dial_indicator(self, text, color):
        """Dairesel gösterge oluştur"""
        dial = QLabel(text)
        dial.setStyleSheet(f"""
            background-color: {self.colors['dark_primary'].name()};
            border-radius: 100px;
            color: {self.colors['text_color'].name()};
            font: bold 24px;
            qproperty-alignment: AlignCenter;
            border: 3px solid {color.name()};
        """)
        dial.setFixedSize(150, 150)
        return dial

    def create_control_panel(self):
        """Kontrol panelini oluştur (güncellenmiş versiyon)"""
        control_frame = QGroupBox("DRONE KONTROLLERİ")
        self.apply_groupbox_style(control_frame)

        control_layout = QGridLayout(control_frame)
        control_layout.setSpacing(8)  # Daha sık aralık
        control_layout.setContentsMargins(10, 15, 10, 10)

        # ARM LED ve buton
        self.arm_led = QLabel()
        self.arm_led.setFixedSize(16, 16)  # Daha küçük LED
        self.arm_btn = self.create_button("ARM / DİSARM", self.arm_drone)
        
        control_layout.addWidget(QLabel("ARM DURUMU:"), 0, 0)
        control_layout.addWidget(self.arm_led, 0, 1)
        control_layout.addWidget(self.arm_btn, 1, 0, 1, 2)

        # Kalkış butonu
        self.takeoff_btn = self.create_button(
            "KALKIŞ YAP",
            self.takeoff_drone,
            color1="#27ae60",
            color2="#2ecc71"
        )
        control_layout.addWidget(self.takeoff_btn, 2, 0, 1, 2)

        # Hedef irtifa input
        control_layout.addWidget(QLabel("HEDEF İRTİFA (m):"), 3, 0)
        self.altitude_input = QLineEdit("3.0")
        self.altitude_input.setValidator(QDoubleValidator(0.5, 20.0, 1))
        self.altitude_input.setStyleSheet(f"""
            QLineEdit {{
                background: {self.colors['dark_primary'].name()};
                color: {self.colors['text_color'].name()};
                border: 1px solid {self.colors['blue_primary'].name()};
                border-radius: 4px;
                padding: 5px;
            }}
        """)
        self.altitude_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.altitude_input, 3, 1)

        # Durum güncellemeleri
        self.update_armed_status(self.armed)
        self.update_connection_status(self.connected)

        return control_frame
    

    def create_button(self, text, callback, enabled=True, color1=None, color2=None):
        """Özelleştirilmiş buton oluştur (küçültülmüş versiyon)"""
        if color1 is None:
            color1 = self.colors['dark_secondary'].name()
        if color2 is None:
            color2 = self.colors['dark_primary'].name()
            
        btn = QPushButton(text)
        btn.setEnabled(enabled)
        btn.setFixedHeight(32)  # Daha küçük buton
        btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {color1}, stop:1 {color2});
                border: 1px solid {self.colors['blue_primary'].name()};
                color: {self.colors['text_color'].name()};
                font: bold 11px;
                padding: 5px 8px;
                border-radius: 4px;
                min-width: 80px;
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {color2}, stop:1 {color1});
            }}
            QPushButton:disabled {{
                background: #95a5a6;
                color: #7f8c8d;
            }}
        """)
        btn.clicked.connect(callback)
        return btn

    def create_console(self):
        """Konsol alanını oluştur"""
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.colors['dark_primary'].name()};
                color: {self.colors['text_color'].name()};
                border: 2px solid {self.colors['dark_secondary'].name()};
                border-radius: 8px;
                padding: 10px;
                font-family: Consolas;
                font-size: 12px;
            }}
        """)
        self.log_console.setMinimumHeight(150)
        self.main_layout.addWidget(self.log_console, 3, 0, 1, 4)

    def create_status_bar(self):
        """Durum çubuğunu oluştur"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background: {self.colors['dark_secondary'].name()};
                color: {self.colors['text_secondary'].name()};
                border-top: 1px solid {self.colors['dark_primary'].name()};
                font-size: 12px;
            }}
        """)
        self.status_bar.showMessage("✅ Sistem hazır | © 2023 Drone Kontrol Arayüzü")
    def update_armed_status(self, armed):
        """ARM durumunu güncelle"""
        self.armed = armed  # ARM durumunu kaydet
        color = self.colors['green_primary'] if armed else self.colors['red_primary']
        self.arm_led.setStyleSheet(f"""
            background-color: {color.name()};
            border-radius: 10px;
            border: 2px solid {self.colors['dark_primary'].name()};
        """)
        # Kalkış butonunu güvenli şekilde güncelle
        if hasattr(self, 'takeoff_btn'):
            self.takeoff_btn.setEnabled(self.armed)
    def apply_groupbox_style(self, groupbox):
        """GroupBox stilini uygula"""
        groupbox.setStyleSheet(f"""
            QGroupBox {{
                color: {self.colors['text_color'].name()};
                font: bold 14px;
                margin-top: 20px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)

    def set_com_port(self, port):
        """COM portunu ayarla"""
        self.selected_com = str(port)  # Tip garantisi ekleyin
        for action in self.com_actions:
            action_text = str(action.text())  # Metni string'e çevirin
            action.setChecked(action_text.startswith(str(port)))  # Her iki tarafı da string yapın
        self.log_message(f"COM port {port} olarak ayarlandı")

    def set_baudrate(self, rate):
        """Baudrate ayarla"""
        self.selected_baud = rate
        for action in self.baud_actions:
            action.setChecked(action.text() == rate)
        self.log_message(f"Baudrate {rate} olarak ayarlandı")

    def connect_drone(self):
        """Drone bağlantısını başlat veya sonlandır"""
        try:
            if not self.connected:
                # Bağlantı öncesi kontroller
                if not self.selected_com:
                    QMessageBox.warning(self, "Hata", "Lütfen önce bir COM port seçin!")
                    return
                
                # Bağlantıyı başlat
                self.log_message(f"{self.selected_com} portuna bağlanılıyor...")
                
                # Bağlantı simülasyonu (gerçek kodda serial.Serial kullanılır)
                self.signals.connection_changed.emit(True)
                self.signals.message_received.emit(f"✅ Bağlantı kuruldu: {self.selected_com} @ {self.selected_baud}")
                
                # Telemetri verilerini simüle et
                self.signals.telemetry_updated.emit({
                    'altitude': "0.0 m",
                    'velocity': "0.0 m/s", 
                    'battery': "100 %",
                    'gps': "0"
                })
                
            else:
                # Bağlantıyı kes
                self.signals.connection_changed.emit(False)
                self.signals.message_received.emit("⏏️ Bağlantı kesildi")
                self.signals.armed_changed.emit(False)  # Bağlantı kesilince ARM'ı da kapat
                
        except Exception as e:
            error_msg = f"Bağlantı hatası: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            QMessageBox.critical(
                self,
                "Bağlantı Hatası",  # Bu satırda eksik string birleştirme vardı
                f"Aşağıdaki hata oluştu:\n{error_msg}\n\n"
                "Lütfen:\n"
                "1. Kabloları kontrol edin\n"
                "2. Doğru port seçtiğinize emin olun\n"
                "3. Cihaz yöneticisinden portun çalıştığını doğrulayın"
            )

    def update_connection_status(self, connected):
        """Bağlantı durumunu güncelle"""
        self.connected = connected
        if connected:
            self.connection_status.setText("🟢 BAĞLANDI")
            self.connection_status.setStyleSheet(f"font: bold 16px; color: {self.colors['green_primary'].name()};")
        else:
            self.connection_status.setText("🔴 BAĞLI DEĞİL")
            self.connection_status.setStyleSheet(f"font: bold 16px; color: {self.colors['red_primary'].name()};")
        # Kalkış butonunu güvenli şekilde güncelle
        if hasattr(self, 'takeoff_btn'):
            self.takeoff_btn.setEnabled(self.armed and self.connected)




    def update_telemetry(self, data):
        """Telemetri verilerini güncelle"""
        self.telemetry_labels['altitude'].setText(f"📏 İRTİFA: {data['altitude']}")
        self.telemetry_labels['velocity'].setText(f"🚀 HIZ: {data['velocity']}")
        self.telemetry_labels['battery'].setText(f"🔋 PİL: {data['battery']}")
        self.telemetry_labels['gps'].setText(f"🛰️ UYDU: {data['gps']}")

    def update_altitude_label(self, value):
        """İrtifa etiketini güncelle"""
        float_value = value / 2
        self.altitude_slider_label.setText(f"HEDEF İRTİFA: {float_value:.1f} m")

    def log_message(self, message):
        """Konsola mesaj yaz"""
        self.log_console.append(message)

    def open_file(self):
        """Dosya açma işlemi"""
        self.log_message("Dosya açma işlemi başlatıldı")

    def save_file(self):
        """Dosya kaydetme işlemi"""
        self.log_message("Dosya kaydetme işlemi başlatıldı")

    def exit_app(self):
        """Uygulamadan çık"""
        self.close()

    def show_about(self):
        """Hakkında bilgisi göster"""
        about = QMessageBox(self)
        about.setWindowTitle("Hakkında")
        about.setText("""
        <h2>Drone Kontrol Arayüzü</h2>
        <p>Versiyon: 1.2.0</p>
        <p>Geliştirici: [Your Name]</p>
        <p>© 2023 Tüm hakları saklıdır.</p>
        """)
        about.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Koyu tema
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    window = DroneInterface()
    #app.aboutToQuit.connect(window.ros_cam_thread.stop)
    window.show()
    sys.exit(app.exec())