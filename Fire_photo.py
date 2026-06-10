import os
import threading
from dotenv import load_dotenv
import sys
import cv2
import datetime
import requests
import time
import json
import socket
import concurrent.futures

ai_lock = threading.Lock()

import onnxruntime
original_session = onnxruntime.InferenceSession

def patched_session(path, providers, **kwargs):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 2
    return original_session(path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'], sess_options=options, **kwargs)

onnxruntime.InferenceSession = patched_session

import ultralytics.utils.checks
original_check = ultralytics.utils.checks.check_requirements

def patched_check(requirements, *args, **kwargs):
    if 'onnxruntime' in str(requirements):
        return
    return original_check(requirements, *args, **kwargs)

ultralytics.utils.checks.check_requirements = patched_check

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Tentukan base path aplikasi (apakah dijalankan sebagai script .py atau .exe hasil compile)
if getattr(sys, 'frozen', False):
    application_path = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    executable_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
    executable_path = os.path.dirname(os.path.abspath(__file__))

# Load variabel dari file .env berdasarkan lokasi resource
dotenv_path = os.path.join(application_path, '.env')
load_dotenv(dotenv_path)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
                             QPushButton, QHBoxLayout, QGridLayout, QFileDialog, QMessageBox,
                             QSizePolicy, QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
                             QComboBox, QMenu, QGroupBox, QSlider, QTreeWidget, QTreeWidgetItem, QHeaderView, QAbstractItemView, QFrame)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtMultimedia import QMediaDevices
from ultralytics import YOLO
from plyer import notification
from playsound import playsound


GLOBAL_STYLE = """
/* Global Window Styles */
QMainWindow {
    background-color: #121212;
}
QDialog {
    background-color: #1E1E1E;
    color: #E0E0E0;
}
QWidget {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

/* Label Styles */
QLabel {
    color: #E0E0E0;
}
QLabel#titleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #FFFFFF;
    margin-bottom: 10px;
}

/* Input Fields (QLineEdit) */
QLineEdit {
    background-color: #2D2D2D;
    color: #FFFFFF;
    border: 1px solid #3D3D3D;
    border-radius: 6px;
    padding: 6px 12px;
}
QLineEdit:hover {
    border: 1px solid #555555;
}
QLineEdit:focus {
    border: 1.5px solid #FF5722; /* Flame Accent */
    background-color: #333333;
}

/* Buttons (QPushButton) */
QPushButton {
    background-color: #2D2D2D;
    color: #E0E0E0;
    border: 1px solid #3D3D3D;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3D3D3D;
    border-color: #555555;
    color: #FFFFFF;
}
QPushButton:pressed {
    background-color: #1E1E1E;
}
QPushButton:disabled {
    background-color: #1E1E1E;
    color: #666666;
    border-color: #2D2D2D;
}

/* Custom Styled Buttons */
QPushButton#primaryBtn {
    background-color: #0078D7;
    border: 1px solid #006CC2;
    padding: 8px 16px;
    color: #FFFFFF;
}
QPushButton#primaryBtn:hover {
    background-color: #1E88E5;
    border-color: #1976D2;
}
QPushButton#primaryBtn:pressed {
    background-color: #0B5CA3;
}

QPushButton#menuBtn {
    background-color: #0078D7;
    border: 1px solid #006CC2;
    color: #FFFFFF;
}
QPushButton#menuBtn:hover {
    background-color: #1E88E5;
    border-color: #1976D2;
}
QPushButton#menuBtn:pressed {
    background-color: #0B5CA3;
}

QPushButton#registerBtn {
    background-color: #2E7D32;
    border: 1px solid #256329;
    padding: 8px 16px;
    color: #FFFFFF;
}
QPushButton#registerBtn:hover {
    background-color: #388E3C;
    border-color: #2E7D32;
}
QPushButton#registerBtn:pressed {
    background-color: #1B5E20;
}

QPushButton#dangerBtn {
    background-color: #C62828;
    border: 1px solid #B71C1C;
    color: #FFFFFF;
}
QPushButton#dangerBtn:hover {
    background-color: #D32F2F;
    border-color: #C62828;
}
QPushButton#dangerBtn:pressed {
    background-color: #8E0000;
}

/* ComboBox Styles */
QComboBox {
    background-color: #2D2D2D;
    color: #E0E0E0;
    border: 1px solid #3D3D3D;
    border-radius: 6px;
    padding: 6px 12px;
}
QComboBox:hover {
    border-color: #555555;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1E1E1E;
    color: #E0E0E0;
    border: 1px solid #3D3D3D;
    selection-background-color: #FF5722;
    selection-color: #FFFFFF;
}

/* Slider Styles */
QSlider::groove:horizontal {
    height: 6px;
    background: #3D3D3D;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #FF5722;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    width: 14px;
    height: 14px;
    margin-top: -4px;
    border-radius: 7px;
    border: 1px solid #CCCCCC;
}
QSlider::handle:horizontal:hover {
    background: #FF5722;
    border-color: #FF7043;
}

/* TreeWidget and Table Headers */
QTreeWidget {
    background-color: #1E1E1E;
    color: #E0E0E0;
    border: 1px solid #3D3D3D;
    border-radius: 6px;
    padding: 5px;
}
QTreeWidget::item {
    padding: 6px;
    border-bottom: 1px solid #252525;
}
QTreeWidget::item:hover {
    background-color: #2D2D2D;
    border-radius: 4px;
}
QTreeWidget::item:selected {
    background-color: #FF5722;
    color: #FFFFFF;
    border-radius: 4px;
}
QHeaderView::section {
    background-color: #2D2D2D;
    color: #FFFFFF;
    padding: 6px;
    border: none;
    font-weight: bold;
    border-bottom: 2px solid #3D3D3D;
}

/* GroupBox Styles */
QGroupBox {
    border: 1px solid #3D3D3D;
    border-radius: 8px;
    margin-top: 20px;
    font-weight: bold;
    color: #FF9900;
    padding-top: 15px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    background-color: #1E1E1E;
}

/* ScrollBar Styles */
QScrollBar:vertical {
    background: #1E1E1E;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #3D3D3D;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #555555;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


# ==========================================
# PEKERJA LATAR BELAKANG (MULTITHREADING)
# ==========================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    alert_signal = pyqtSignal(str)
    fps_signal = pyqtSignal(str)

    def __init__(self, source, model_path, alarm_path, room_name, conf_fire=0.8, conf_smoke=0.5):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.alarm_path = alarm_path
        self.ROOM_NAME = room_name
        self.conf_fire = conf_fire
        self.conf_smoke = conf_smoke
        self.conf_threshold = min(conf_fire, conf_smoke)
        self.API_URL = os.getenv('URL_UPLOAD')
        self.last_fire_alert_time = 0
        self.last_smoke_alert_time = 0
        self.alert_cooldown = 15
        self._run_flag = True
        self.latest_frame = None
        self.frame_ready = False

    def grab_frames(self):
        # 1. Paksa OpenCV menggunakan mesin pembaca FFmpeg (cv2.CAP_FFMPEG)
        # 2. Tangkap error jika source-nya berupa integer (Kamera USB lokal)
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self._run_flag:
            if not self.cap.isOpened():
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
                self.frame_ready = True
                time.sleep(0.015)
            else:
                self.frame_ready = False
                if isinstance(self.source, str):
                    if self.source.lower().startswith("rtsp"):
                        print(f"[{self.ROOM_NAME}] Koneksi RTSP Lemah/Terputus! Reconnect dalam 3 detik...")
                        self.cap.release()
                        time.sleep(3)
                        # --- PASTIKAN RECONNECT JUGA MENGGUNAKAN FFMPEG ---
                        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    else:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    break

        if self.cap is not None:
            self.cap.release()

    def run(self):
        # Setiap kamera memuat modelnya sendiri agar tracker state (ByteTrack) terisolasi dan tidak bercampur
        self.model = YOLO(self.model_path, task='detect')

        grabber_thread = threading.Thread(target=self.grab_frames, daemon=True)
        grabber_thread.start()

        last_time = time.time()
        last_ai_run = 0
        ai_cooldown = 1.0
        saved_fire_boxes = []

        while self._run_flag:
            if not self.frame_ready or self.latest_frame is None:
                time.sleep(0.01)
                continue

            frame = self.latest_frame.copy()
            self.frame_ready = False

            try:
                current_time_ai = time.time()

                if current_time_ai - last_ai_run >= ai_cooldown:
                    last_ai_run = current_time_ai

                    # Set threshold terkecil agar model mendeteksi semua candidate
                    self.conf_threshold = min(self.conf_fire, self.conf_smoke)

                    with ai_lock:
                        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640, conf=self.conf_threshold, verbose=False)

                    saved_display_boxes = []
                    has_fire = False
                    has_smoke = False
                    fire_conf = 0
                    smoke_conf = 0

                    if len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf_value = round(float(box.conf[0]) * 100, 2)
                            
                            # Filter berdasarkan sensitivitas spesifik per kelas
                            if cls_id == 0:
                                if conf_value >= (self.conf_fire * 100):
                                    has_fire = True
                                    fire_conf = max(fire_conf, conf_value)
                                    saved_display_boxes.append((x1, y1, x2, y2, conf_value, cls_id))
                            elif cls_id == 1:
                                if conf_value >= (self.conf_smoke * 100):
                                    has_smoke = True
                                    smoke_conf = max(smoke_conf, conf_value)
                                    saved_display_boxes.append((x1, y1, x2, y2, conf_value, cls_id))

                    if has_fire or has_smoke:
                        # Gambar bounding box yang lolos filter ke screenshot
                        annotated_frame = frame.copy()
                        for (x1, y1, x2, y2, conf_val, c_id) in saved_display_boxes:
                            if c_id == 0:
                                color = (0, 0, 255) # Red for fire
                                label = f"Fire: {conf_val}%"
                            else:
                                color = (255, 165, 0) # Orange for smoke
                                label = f"Smoke: {conf_val}%"
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        if has_fire and has_smoke:
                            if (current_time_ai - self.last_fire_alert_time > self.alert_cooldown) or \
                               (current_time_ai - self.last_smoke_alert_time > self.alert_cooldown):
                                self.last_fire_alert_time = current_time_ai
                                self.last_smoke_alert_time = current_time_ai
                                self.alert_signal.emit(f"{self.ROOM_NAME} (Kebakaran & Asap)")
                                
                                conf_value = max(fire_conf, smoke_conf)
                                threading.Thread(target=self.trigger_actions,
                                                 args=(annotated_frame.copy(), conf_value, True, "fire + smoke"), daemon=True).start()
                        elif has_fire:
                            if current_time_ai - self.last_fire_alert_time > self.alert_cooldown:
                                self.last_fire_alert_time = current_time_ai
                                self.alert_signal.emit(f"{self.ROOM_NAME} (Kebakaran)")
                                
                                threading.Thread(target=self.trigger_actions,
                                                 args=(annotated_frame.copy(), fire_conf, True, "fire"), daemon=True).start()
                        elif has_smoke:
                            if current_time_ai - self.last_smoke_alert_time > self.alert_cooldown:
                                self.last_smoke_alert_time = current_time_ai
                                self.alert_signal.emit(f"{self.ROOM_NAME} (Asap)")
                                
                                threading.Thread(target=self.trigger_actions,
                                                 args=(annotated_frame.copy(), smoke_conf, False, "smoke"), daemon=True).start()

                display_frame = frame.copy()

                for (x1, y1, x2, y2, conf_value, cls_id) in saved_display_boxes:
                    if cls_id == 0:
                        color = (0, 0, 255) # Red for fire
                        label = f"Fire: {conf_value}%"
                    else:
                        color = (255, 165, 0) # Orange for smoke
                        label = f"Smoke: {conf_value}%"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(q_img)

                current_time = time.time()
                elapsed_time = current_time - last_time
                if elapsed_time > 0:
                    video_fps = 1.0 / elapsed_time
                    self.fps_signal.emit(f"⚡ Video: {video_fps:.0f} FPS")
                last_time = current_time

                time.sleep(0.04)

            except Exception as e:
                pass

    def stop(self):
        self._run_flag = False
        self.wait()

    def trigger_actions(self, frame, confidence, play_alarm=True, label_type="fire"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label_type}_{self.ROOM_NAME}_{timestamp}_{confidence}.jpg"

        if play_alarm and self.alarm_path and os.path.exists(self.alarm_path):
            try:
                playsound(self.alarm_path, block=False)
            except Exception as e:
                pass

        fire_events_dir = os.path.join(executable_path, "fire_events")
        os.makedirs(fire_events_dir, exist_ok=True)
        local_path = os.path.join(fire_events_dir, filename)
        cv2.imwrite(local_path, frame)

        try:
            with open(local_path, 'rb') as f:
                kunci_rahasia = os.getenv("API_KEY")
                headers = {'x-api-key': kunci_rahasia}
                requests.post(self.API_URL,
                              files={'photo': (filename, f, 'image/jpeg')},
                              data={'room': self.ROOM_NAME, 'timestamp': timestamp, 'confidence': confidence, 'status': label_type},
                              headers=headers,
                              timeout=3)
        except Exception as e:
            print(f"API Offline: {e}")


# ==========================================
# WIDGET UNTUK SATU LAYAR KAMERA/VIDEO
# ==========================================
class CameraWidget(QWidget):
    def __init__(self, source, model_path, alarm_path, room_name="Ruangan", conf_fire=0.8, conf_smoke=0.5, remove_callback=None):
        super().__init__()
        self.remove_callback = remove_callback
        self.source = source
        self.ROOM_NAME = room_name

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.header_layout = QHBoxLayout()
        self.title_label = QLabel(self.ROOM_NAME)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #FFFFFF;")

        self.fps_label = QLabel("⚡ Video: -- FPS")
        self.fps_label.setStyleSheet(
            "color: #00ff00; font-family: monospace; font-weight: bold; background-color: #1A1A1A; padding: 4px 8px; border-radius: 4px; border: 1px solid #2D2D2D;")

        self.btn_close = QPushButton("❌ Tutup")
        self.btn_close.setObjectName("dangerBtn")
        self.btn_close.setFixedWidth(80)
        self.btn_close.clicked.connect(self.close_camera)

        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.fps_label)
        self.header_layout.addSpacing(10)
        self.header_layout.addWidget(self.btn_close)
        self.layout.addLayout(self.header_layout)

        self.label = QLabel(f"Memuat {self.ROOM_NAME}...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("border: 1px solid #3D3D3D; border-radius: 6px; background-color: #000000; color: #888888;")
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.label.setMinimumSize(320, 240)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.video_thread = VideoThread(source, model_path, alarm_path, self.ROOM_NAME, conf_fire, conf_smoke)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.alert_signal.connect(self.show_windows_alert)
        self.video_thread.fps_signal.connect(self.update_fps)
        self.video_thread.start()

    def show_windows_alert(self, alert_msg):
        try:
            if "(Asap)" in alert_msg:
                title = "PERINGATAN!"
                msg = f"Terdeteksi Asap di {alert_msg.replace(' (Asap)', '')}"
            else:
                title = "BAHAYA!"
                msg = f"Terdeteksi Api di {alert_msg.replace(' (Kebakaran)', '')}"
            notification.notify(title=title, message=msg, timeout=5)
        except Exception as e:
            pass

    def update_image(self, q_img):
        self.label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def update_fps(self, fps_str):
        self.fps_label.setText(fps_str)

    def close_camera(self):
        self.video_thread.stop()
        if self.remove_callback:
            self.remove_callback(self)
        self.setParent(None)
        self.deleteLater()


# ==========================================
# THREAD PEMINDAI JARINGAN (Versi Cepat & Paralel)
# ==========================================
class ScannerThread(QThread):
    result_signal = pyqtSignal(list)
    progress_signal = pyqtSignal(str)

    def run(self):
        self.progress_signal.emit("Mencari IP lokal...")
        local_ip = self.get_local_ip()
        base_ip = ".".join(local_ip.split(".")[:-1]) + "."
        discovered_ips = []
        self.progress_signal.emit(f"Memindai {base_ip}x secara paralel...")

        def scan_ip(target_ip):
            if self.check_port(target_ip, 554, timeout=0.5): return target_ip
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            ips_to_scan = [f"{base_ip}{i}" for i in range(1, 255)]
            results = executor.map(scan_ip, ips_to_scan)
            for res in results:
                if res:
                    discovered_ips.append(res)
                    self.progress_signal.emit(f"Ditemukan CCTV di: {res}")

        self.result_signal.emit(discovered_ips)

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "192.168.1.1"

    def check_port(self, ip, port, timeout=0.5):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False


# ==========================================
# DIALOG TAMBAH KAMERA (Versi Auto-Discovery)
# ==========================================
class AddCameraDialog(QDialog):
    def __init__(self, active_ports, parent=None):
        super().__init__(parent)
        self.active_ports = active_ports
        self.setWindowTitle("Tambah Kamera Pintar")
        self.setMinimumWidth(450)
        self.layout = QVBoxLayout(self)

        form_layout = QFormLayout()
        self.room_input = QLineEdit(self)
        self.room_input.setPlaceholderText("Misal: Dapur")
        form_layout.addRow("Nama Ruangan:", self.room_input)

        self.source_input = QComboBox(self)
        form_layout.addRow("Kamera USB:", self.source_input)
        self.populate_ports()
        self.layout.addLayout(form_layout)

        cctv_group = QGroupBox("Otomatisasi CCTV (IP Camera)")
        cctv_layout = QFormLayout()

        scan_layout = QHBoxLayout()
        self.btn_scan = QPushButton("🔍 Scan Jaringan")
        self.btn_scan.clicked.connect(self.start_scan)

        self.cctv_dropdown = QComboBox()
        self.cctv_dropdown.addItem("-- Klik Scan Jaringan --")

        scan_layout.addWidget(self.btn_scan)
        scan_layout.addWidget(self.cctv_dropdown)
        cctv_layout.addRow("Pilih CCTV:", scan_layout)

        self.user_input = QLineEdit()
        self.user_input.setText("admin")
        cctv_layout.addRow("Username:", self.user_input)

        self.pass_input = QLineEdit()
        self.pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pass_input.setPlaceholderText("Kode Verifikasi (Misal: QRRTSQ)")
        cctv_layout.addRow("Password:", self.pass_input)

        self.stream_input = QComboBox()
        self.stream_input.addItems(["sub (Lancar untuk AI)", "main (Kualitas HD)"])
        cctv_layout.addRow("Kualitas:", self.stream_input)

        cctv_group.setLayout(cctv_layout)
        self.layout.addWidget(cctv_group)

        manual_layout = QFormLayout()
        self.manual_rtsp = QLineEdit()
        self.manual_rtsp.setPlaceholderText("Atau paste link RTSP manual di sini...")
        manual_layout.addRow("Manual Link:", self.manual_rtsp)
        self.layout.addLayout(manual_layout)

        self.buttons = QDialogButtonBox()
        self.buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        self.buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def populate_ports(self):
        video_devices = QMediaDevices.videoInputs()
        self.source_input.addItem("-- Pilih Kamera USB --", None)
        for i, device in enumerate(video_devices):
            if i not in self.active_ports:
                camera_name = device.description()
                self.source_input.addItem(f"{camera_name} (Port {i})", i)

    def start_scan(self):
        self.btn_scan.setEnabled(False)
        self.cctv_dropdown.clear()
        self.cctv_dropdown.addItem("Memindai... (Tunggu sebentar)")

        self.scanner = ScannerThread()
        self.scanner.progress_signal.connect(self.update_scan_progress)
        self.scanner.result_signal.connect(self.scan_finished)
        self.scanner.start()

    def update_scan_progress(self, msg):
        self.cctv_dropdown.setItemText(0, msg)

    def scan_finished(self, ips):
        self.cctv_dropdown.clear()
        if ips:
            for ip in ips:
                self.cctv_dropdown.addItem(f"CCTV Ditemukan: {ip}", ip)
        else:
            self.cctv_dropdown.addItem("Tidak ada CCTV ditemukan", None)
        self.btn_scan.setEnabled(True)

    def get_data(self):
        room_name = self.room_input.text().strip()
        manual_url = self.manual_rtsp.text().strip()
        if manual_url: return room_name, manual_url
        selected_ip = self.cctv_dropdown.currentData()
        if selected_ip:
            user = self.user_input.text().strip()
            password = self.pass_input.text().strip()
            stream = self.stream_input.currentText().split()[0]
            constructed_rtsp = f"rtsp://{user}:{password}@{selected_ip}:554/ch1/{stream}"
            return room_name, constructed_rtsp
        return room_name, self.source_input.currentData()


# ==========================================
# DIALOG RIWAYAT DETEKSI (TREE VIEW / GROUPING)
# ==========================================
class DatabaseViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Riwayat Deteksi Kebakaran (Grouped View)")
        self.resize(1000, 600)

        layout = QHBoxLayout(self)

        left_panel = QVBoxLayout()

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Kategori / Nama File", "Waktu Kejadian", "Lokasi", "Akurasi (%)"])

        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree.itemSelectionChanged.connect(self.display_selected_image)

        self.btn_refresh = QPushButton("🔄 Refresh Data")
        self.btn_refresh.setObjectName("primaryBtn")
        self.btn_refresh.clicked.connect(self.load_data)

        self.lbl_server_status = QLabel("Memeriksa data...")
        self.lbl_server_status.setStyleSheet("font-weight: bold; color: #888888;")

        left_panel.addWidget(self.tree)
        left_panel.addWidget(self.lbl_server_status)
        left_panel.addWidget(self.btn_refresh)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #2D2D2D;")

        right_panel = QVBoxLayout()
        self.image_preview = QLabel("Pilih data di daftar\nuntuk melihat foto kejadian")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setStyleSheet("border: 2px dashed #3D3D3D; border-radius: 8px; background-color: #121212; color: #888888; font-size: 14px;")
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_preview.setMinimumSize(450, 450)

        self.lbl_detail = QLabel("-")
        self.lbl_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_detail.setStyleSheet("font-weight: bold; color: #ff5722; padding: 8px; background-color: #1A1A1A; border-radius: 6px; border: 1px solid #2D2D2D;")

        right_panel.addWidget(self.image_preview)
        right_panel.addWidget(self.lbl_detail)

        layout.addLayout(left_panel, 1)
        layout.addWidget(separator)
        layout.addLayout(right_panel, 1)

        self.load_data()

    def load_data(self):
        self.tree.clear()

        root_local = QTreeWidgetItem(self.tree)
        root_local.setText(0, "📁 Data Lokal (Penyimpanan Komputer)")
        root_local.setFirstColumnSpanned(True)

        root_server = QTreeWidgetItem(self.tree)
        root_server.setText(0, "🌐 Data Server (Database MySQL)")
        root_server.setFirstColumnSpanned(True)

        API_URL_GET = "http://127.0.0.1:5000/get_history"

        try:
            kunci_rahasia = os.getenv("API_KEY")
            headers = {'x-api-key': kunci_rahasia}
            response = requests.get(API_URL_GET, headers=headers, timeout=2)

            if response.status_code == 200:
                data_db = response.json()
                for data in data_db:
                    file_name = os.path.basename(str(data['photos']))

                    child = QTreeWidgetItem(root_server)
                    child.setText(0, file_name)
                    child.setText(1, str(data['waktu_kejadian']))
                    child.setText(2, str(data['location']))
                    child.setText(3, f"{data['confidence_score']} %")

                self.lbl_server_status.setText("🟢 Server API: ONLINE (Data Sinkron)")
                self.lbl_server_status.setStyleSheet("font-weight: bold; color: #00ff00;")
                root_server.setExpanded(True)
            else:
                raise Exception("API Bukan 200")

        except Exception as e:
            root_server.setHidden(True)
            self.lbl_server_status.setText("🔴 Server API: OFFLINE (Hanya Menampilkan Lokal)")
            self.lbl_server_status.setStyleSheet("font-weight: bold; color: #ff4c4c;")

        folder_path = os.path.join(executable_path, "fire_events")
        if os.path.exists(folder_path):
            files = sorted(os.listdir(folder_path), reverse=True)
            for file in files:
                if file.endswith((".jpg", ".png")):
                    parts = file.replace(".jpg", "").replace(".png", "").split("_")
                    ruangan = parts[1] if len(parts) > 1 else "Unknown"

                    if len(parts) >= 5:
                        tgl, jam, akurasi = parts[2], parts[3], parts[4]
                    elif len(parts) == 4:
                        tgl, jam, akurasi = parts[2], parts[3], "N/A"
                    else:
                        tgl, jam, akurasi = "Unknown", "Unknown", "N/A"

                    try:
                        waktu = f"{tgl[:4]}-{tgl[4:6]}-{tgl[6:]} {jam[:2]}:{jam[2:4]}:{jam[4:]}"
                    except:
                        waktu = f"{tgl}_{jam}"

                    child = QTreeWidgetItem(root_local)
                    child.setText(0, file)
                    child.setText(1, waktu)
                    child.setText(2, ruangan)
                    child.setText(3, f"{akurasi} %" if akurasi != "N/A" else "N/A")

                    if akurasi == "N/A":
                        child.setForeground(3, Qt.GlobalColor.gray)

            root_local.setExpanded(True)

    def display_selected_image(self):
        selected_items = self.tree.selectedItems()
        if not selected_items: return

        item = selected_items[0]

        if item.parent() is None:
            return

        file_name = item.text(0)
        waktu = item.text(1)
        ruangan = item.text(2)
        akurasi = item.text(3)

        path_gui = os.path.join(executable_path, "fire_events", file_name)
        path_api_local = os.path.join(executable_path, "api_uploads", file_name)
        path_api_shared = os.path.abspath(os.path.join(executable_path, "..", "api_server", "api_uploads", file_name))

        if os.path.exists(path_gui):
            image_path = path_gui
        elif os.path.exists(path_api_local):
            image_path = path_api_local
        elif os.path.exists(path_api_shared):
            image_path = path_api_shared
        else:
            image_path = None

        if image_path:
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(self.image_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.image_preview.setPixmap(scaled_pixmap)
            self.lbl_detail.setText(f"📍 {ruangan} | ⏰ {waktu} | 🎯 {akurasi}")
            self.lbl_detail.setStyleSheet("font-weight: bold; color: #00ff00; padding: 5px;")
        else:
            self.image_preview.setText("❌ Gambar asli terhapus atau tidak ditemukan.")
            self.image_preview.setPixmap(QPixmap())
            self.lbl_detail.setText(f"Data gambar {file_name} hilang.")
            self.lbl_detail.setStyleSheet("font-weight: bold; color: #ff4c4c; padding: 5px;")

# ==========================================
# JENDELA UTAMA
# ==========================================
class FireDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem Deteksi Kebakaran YOLO11")
        self.setGeometry(100, 100, 1000, 700)

        self.config_file = os.path.join(executable_path, "camera_config.json")
        self.settings_file = os.path.join(executable_path, "settings.json")
        self.load_app_settings()

        if not os.path.exists(self.model_path):
            QMessageBox.critical(None, "Fatal Error - Model Hilang",
                                 f"File model AI tidak ditemukan di lokasi:\n{self.model_path}\n\nSistem ditutup.")
            sys.exit()

        if not os.path.exists(self.alarm_path):
            self.alarm_path = ""

        try:
            self.model = YOLO(self.model_path, task='detect')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal memuat model YOLO: {e}")
            sys.exit()

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Styled Toolbar Header
        self.header_frame = QFrame()
        self.header_frame.setObjectName("headerFrame")
        self.header_layout = QHBoxLayout(self.header_frame)
        self.header_layout.setContentsMargins(15, 10, 15, 10)

        self.btn_menu = QPushButton(" Menu Utama")
        self.btn_menu.setObjectName("menuBtn")
        self.btn_menu.setFixedWidth(130)

        menu = QMenu(self)
        menu.addAction("  Tambah Kamera", self.add_camera)
        menu.addAction("  Upload Video (Test)", self.upload_video)
        menu.addAction("  History Kebakaran", self.open_database_viewer)
        menu.addSeparator()
        menu.addAction("  Update Model AI (.onnx / .pt)", self.change_model)
        menu.addAction("  Ganti Suara Alarm (.mp3 / .wav)", self.change_alarm)
        self.btn_menu.setMenu(menu)

        env_val = os.getenv('CONFIDENCE_INITIAL_VALUE')
        try:
            initial_value_fire = int(env_val) if env_val else 80
        except ValueError:
            initial_value_fire = 80
        
        initial_value_smoke = 50

        self.current_conf_fire = initial_value_fire / 100.0
        self.current_conf_smoke = initial_value_smoke / 100.0

        # Sensitivitas Kebakaran
        self.slider_label_fire = QLabel(f"Sensitivitas Api: {initial_value_fire}%")
        self.slider_label_fire.setStyleSheet("font-weight: bold; color: #ff3333;")

        self.conf_slider_fire = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider_fire.setRange(10, 100)
        self.conf_slider_fire.setValue(initial_value_fire)
        self.conf_slider_fire.setFixedWidth(120)
        self.conf_slider_fire.valueChanged.connect(self.change_confidence_fire)

        # Sensitivitas Asap
        self.slider_label_smoke = QLabel(f"Sensitivitas Asap: {initial_value_smoke}%")
        self.slider_label_smoke.setStyleSheet("font-weight: bold; color: #ffaa00;")

        self.conf_slider_smoke = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider_smoke.setRange(10, 100)
        self.conf_slider_smoke.setValue(initial_value_smoke)
        self.conf_slider_smoke.setFixedWidth(120)
        self.conf_slider_smoke.valueChanged.connect(self.change_confidence_smoke)

        model_name = os.path.basename(self.model_path)
        alarm_name = os.path.basename(self.alarm_path) if self.alarm_path else "Tidak Ada (Bisu)"
        self.status_label = QLabel(f"Model Aktif: {model_name} | Alarm: {alarm_name}")
        self.status_label.setStyleSheet("color: #888888; font-style: italic;")

        self.header_layout.addWidget(self.btn_menu)
        self.header_layout.addSpacing(20)
        self.header_layout.addWidget(self.slider_label_fire)
        self.header_layout.addWidget(self.conf_slider_fire)
        self.header_layout.addSpacing(15)
        self.header_layout.addWidget(self.slider_label_smoke)
        self.header_layout.addWidget(self.conf_slider_smoke)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.status_label)

        self.main_layout.addWidget(self.header_frame)

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(15)
        self.grid_layout.setContentsMargins(15, 15, 15, 15)

        self.grid_container = QWidget()
        self.grid_container.setLayout(self.grid_layout)
        self.main_layout.addWidget(self.grid_container, 1)

        self.main_layout.setStretch(0, 0)
        self.main_layout.setStretch(1, 1)

        self.cameras = []
        self.load_saved_cameras()

    def rearrange_grid(self):
        for cam in self.cameras:
            self.grid_layout.removeWidget(cam)
        for idx, cam in enumerate(self.cameras):
            row = idx // 2
            col = idx % 2
            self.grid_layout.addWidget(cam, row, col)

    def load_app_settings(self):
        self.model_path = os.path.join(application_path, 'best_AdamW_50_16.onnx')
        self.alarm_path = os.path.join(application_path, 'alarm.mp3')
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.model_path = settings.get('model_path', self.model_path)
                    self.alarm_path = settings.get('alarm_path', self.alarm_path)
            except Exception as e:
                pass

    def save_app_settings(self):
        settings = {'model_path': self.model_path, 'alarm_path': self.alarm_path}
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            pass

    def change_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Model AI", "", "YOLO Models (*.onnx *.pt)")
        if file_name:
            try:
                new_model = YOLO(file_name, task='detect')
                self.model_path = file_name
                self.model = new_model
                for cam in self.cameras: cam.video_thread.model = new_model
                self.save_app_settings()
                self.status_label.setText(
                    f"Model Aktif: {os.path.basename(self.model_path)} | Alarm: {os.path.basename(self.alarm_path) if self.alarm_path else 'Bisu'}")
                QMessageBox.information(self, "Sukses", "Model AI diperbarui!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Model rusak:\n{e}")

    def change_alarm(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Suara Alarm", "", "Audio Files (*.mp3 *.wav)")
        if file_name:
            self.alarm_path = file_name
            for cam in self.cameras: cam.video_thread.alarm_path = file_name
            self.save_app_settings()
            self.status_label.setText(
                f"Model Aktif: {os.path.basename(self.model_path)} | Alarm: {os.path.basename(self.alarm_path)}")
            QMessageBox.information(self, "Sukses", "Alarm diperbarui!")

    def load_saved_cameras(self):
        self.saved_cams_queue = []
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.saved_cams_queue = json.load(f)
            except Exception as e:
                pass

        if self.saved_cams_queue:
            self.loading_timer = QTimer(self)
            self.loading_timer.timeout.connect(self._process_camera_queue)
            self.loading_timer.start(800)
        else:
            self.add_video_source(0, "Kamera Utama")

    def _process_camera_queue(self):
        if self.saved_cams_queue:
            cam_data = self.saved_cams_queue.pop(0)
            self.add_video_source(cam_data['source'], cam_data['name'])
        else:
            self.loading_timer.stop()

    def closeEvent(self, event):
        saved_cams = []
        for cam in self.cameras:
            if cam.isVisible():
                try:
                    saved_cams.append({"name": cam.ROOM_NAME, "source": cam.source})
                except AttributeError:
                    pass
        try:
            with open(self.config_file, 'w') as f:
                json.dump(saved_cams, f, indent=4)
        except Exception:
            pass
        event.accept()

    @staticmethod
    def open_screenshot_folder():
        folder_path = os.path.join(executable_path, "fire_events")
        os.makedirs(folder_path, exist_ok=True)
        os.startfile(folder_path)

    def open_database_viewer(self):
        dialog = DatabaseViewerDialog(self)
        dialog.exec()

    def add_camera(self):
        active_ports = [cam.source for cam in self.cameras if isinstance(cam.source, int)]
        dialog = AddCameraDialog(active_ports, self)
        if dialog.exec():
            room_name, source_index = dialog.get_data()
            if not room_name:
                QMessageBox.warning(self, "Peringatan", "Nama ruangan harus diisi!")
            elif source_index is None:
                QMessageBox.warning(self, "Peringatan", "Kamera belum dipilih!")
            else:
                self.add_video_source(source_index, room_name)

    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Pilih Video Testing", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_name:
            self.add_video_source(file_name, f"Video: {os.path.basename(file_name)}")

    def change_confidence_fire(self, value):
        self.current_conf_fire = value / 100.0
        self.slider_label_fire.setText(f"Sensitivitas Api: {value}%")
        for cam in self.cameras:
            try:
                cam.video_thread.conf_fire = self.current_conf_fire
                cam.video_thread.conf_threshold = min(self.current_conf_fire, cam.video_thread.conf_smoke)
            except AttributeError:
                pass

    def change_confidence_smoke(self, value):
        self.current_conf_smoke = value / 100.0
        self.slider_label_smoke.setText(f"Sensitivitas Asap: {value}%")
        for cam in self.cameras:
            try:
                cam.video_thread.conf_smoke = self.current_conf_smoke
                cam.video_thread.conf_threshold = min(cam.video_thread.conf_fire, self.current_conf_smoke)
            except AttributeError:
                pass

    def remove_camera_from_list(self, cam_widget):
        if cam_widget in self.cameras:
            self.cameras.remove(cam_widget)
            self.rearrange_grid()

    def add_video_source(self, source, name):
        cam_widget = CameraWidget(source, self.model_path, self.alarm_path, name, 
                                  self.current_conf_fire, self.current_conf_smoke,
                                  self.remove_camera_from_list)
        self.cameras.append(cam_widget)
        self.rearrange_grid()


class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem Keamanan - Login")
        self.setFixedSize(360, 240)

        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        self.lbl_title = QLabel("🛡️ Silakan Login")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setObjectName("titleLabel")
        layout.addWidget(self.lbl_title)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        self.inp_username = QLineEdit()
        self.inp_username.setPlaceholderText("Masukkan Username")

        self.inp_password = QLineEdit()
        self.inp_password.setEchoMode(QLineEdit.EchoMode.Password)  # Menyamarkan password (***)
        self.inp_password.setPlaceholderText("Masukkan Password")

        form_layout.addRow("Username:", self.inp_username)
        form_layout.addRow("Password:", self.inp_password)
        layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.btn_register = QPushButton("Daftar Akun")
        self.btn_register.setObjectName("registerBtn")
        self.btn_register.clicked.connect(self.open_register_window)

        self.btn_login = QPushButton(" Login")
        self.btn_login.setObjectName("primaryBtn")
        self.btn_login.clicked.connect(self.attempt_login)

        button_layout.addWidget(self.btn_register)
        button_layout.addWidget(self.btn_login)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def open_register_window(self):
        reg_dialog = RegisterWindow(self)
        reg_dialog.exec()

    def attempt_login(self):
        username = self.inp_username.text().strip()
        password = self.inp_password.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "Peringatan", "Username dan Password tidak boleh kosong!")
            return

        self.btn_login.setEnabled(False)
        self.btn_login.setText("Mengecek ke Server...")
        QApplication.processEvents()

        try:
            # Mengambil alamat API dari .env, lalu mengganti /upload menjadi /login
            api_url = os.getenv('URL_UPLOAD')
            if api_url:
                login_url = api_url.replace('/upload', '/login')
            else:
                login_url = "http://localhost:5000/login"  # Default darurat

            # Mengirim permintaan login ke Backend
            response = requests.post(login_url, json={"username": username, "password": password}, timeout=3)

            if response.status_code == 200:
                QMessageBox.information(self, "Sukses", "Login Berhasil!\nSistem Utama akan dibuka.")
                self.accept()  # Menutup dialog dan memberikan lampu hijau ke aplikasi utama
            else:
                data = response.json()
                QMessageBox.warning(self, "Gagal", data.get("message", "Username atau Password salah!"))

        except requests.exceptions.RequestException:
            QMessageBox.critical(self, "API Offline",
                                 "Tidak dapat terhubung ke Server API!\nPastikan api_server.exe sudah menyala.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Terjadi kesalahan sistem:\n{e}")
        finally:
            self.btn_login.setEnabled(True)
            self.btn_login.setText(" Login")


class RegisterWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sistem Keamanan - Registrasi")
        self.setFixedSize(360, 280)

        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        self.lbl_title = QLabel("📝 Daftar Akun Baru")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setObjectName("titleLabel")
        layout.addWidget(self.lbl_title)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        self.inp_username = QLineEdit()
        self.inp_username.setPlaceholderText("Buat Username Baru")

        self.inp_password = QLineEdit()
        self.inp_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.inp_password.setPlaceholderText("Buat Password")

        self.inp_confirm = QLineEdit()
        self.inp_confirm.setEchoMode(QLineEdit.EchoMode.Password)
        self.inp_confirm.setPlaceholderText("Ulangi Password")

        form_layout.addRow("Username:", self.inp_username)
        form_layout.addRow("Password:", self.inp_password)
        form_layout.addRow("Ulangi Password:", self.inp_confirm)
        layout.addLayout(form_layout)

        self.btn_register = QPushButton(" Daftar Sekarang")
        self.btn_register.setObjectName("registerBtn")
        self.btn_register.clicked.connect(self.attempt_register)
        layout.addWidget(self.btn_register)

        self.setLayout(layout)

    def attempt_register(self):
        username = self.inp_username.text().strip()
        password = self.inp_password.text().strip()
        confirm = self.inp_confirm.text().strip()

        if not username or not password or not confirm:
            QMessageBox.warning(self, "Peringatan", "Semua kolom harus diisi!")
            return

        if password != confirm:
            QMessageBox.warning(self, "Peringatan", "Password dan Ulangi Password tidak cocok!")
            return

        self.btn_register.setEnabled(False)
        self.btn_register.setText("Memproses...")
        QApplication.processEvents()

        try:
            api_url = os.getenv('URL_UPLOAD')
            if api_url:
                register_url = api_url.replace('/upload', '/register')
            else:
                register_url = "http://localhost:5000/register"

            response = requests.post(register_url, json={"username": username, "password": password}, timeout=3)

            if response.status_code == 201:
                QMessageBox.information(self, "Sukses",
                                        "Akun berhasil dibuat!\nSilakan login menggunakan akun baru Anda.")
                self.accept()  # Tutup dialog registrasi
            else:
                data = response.json()
                QMessageBox.warning(self, "Gagal", data.get("message", "Gagal mendaftar!"))

        except requests.exceptions.RequestException:
            QMessageBox.critical(self, "API Offline",
                                 "Tidak dapat terhubung ke Server API!\nPastikan api_server.exe sudah menyala.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Terjadi kesalahan sistem:\n{e}")
        finally:
            self.btn_register.setEnabled(True)
            self.btn_register.setText(" Daftar Sekarang")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLE)

    login_dialog = LoginWindow()

    if login_dialog.exec() == QDialog.DialogCode.Accepted:
        window = FireDetectionApp()
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit()