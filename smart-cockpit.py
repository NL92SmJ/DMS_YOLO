import csv
from datetime import datetime
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)
from ultralytics import YOLO

# --- MULTITHREADED VIDEO STREAM ---
class VideoCaptureAsync:
    """Asynchroner Video-Stream zur Vermeidung von I/O-Blocking und Maximierung der FPS."""
    def __init__(self, src=0, width=1280, height=720):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.grabbed, self.frame = self.cap.read()
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# --- INITIALISIERUNG ---
print("Lade YOLO-Modell (ONNX) & MediaPipe...")
model = YOLO("yolo26n.onnx", task="detect")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# NVML Initialisierung mit Fehlertoleranz für RTX 3050
nvml_active = True
try:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    print(f"Warnung: NVML konnte nicht initialisiert werden: {e}")
    nvml_active = False

# Kamera starten
cap = VideoCaptureAsync(src=0, width=1280, height=720)
cap.start()

cv2.namedWindow("Smart Cockpit - DMS & YOLO", cv2.WINDOW_NORMAL)

# --- CSV-LOGGING SETUP ---
csv_filename = f"smart_cockpit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Timestamp",
    "Yaw",
    "Pitch",
    "Roll",
    "EAR",
    "MAR",
    "Head_State",
    "Blink_Status",
    "Mouth_Status",
    "Detected_Objects",
    "GPU_Util",
    "VRAM_MB",
    "Power_W",
    "FPS",
])
print(f"Logging aktiv in: {csv_filename}")

# Hilfsfunktionen
def calculate_ear(eye_points, landmarks, img_w, img_h):
    p = np.array(
        [(landmarks[idx].x * img_w, landmarks[idx].y * img_h, landmarks[idx].z) for idx in eye_points],
        dtype=np.float64,
    )
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return 0.0 if h == 0 else (v1 + v2) / (2.0 * h)

def calculate_mar(mouth_points, landmarks, img_w, img_h):
    p = np.array(
        [(landmarks[idx].x * img_w, landmarks[idx].y * img_h, landmarks[idx].z) for idx in mouth_points],
        dtype=np.float64,
    )
    v = np.linalg.norm(p[1] - p[0])
    h = np.linalg.norm(p[3] - p[2])
    return 0.0 if h == 0 else v / h

# Schwellenwerte & Konstanten
EAR_THRESHOLD = 0.22
MAR_YAWN_THRESHOLD = 0.45
alpha = 0.3
smooth_yaw, smooth_pitch, smooth_roll = 0.0, 0.0, 0.0
eyes_closed_start_time = None

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 61, 291]
landmark_indices = [1, 199, 33, 263, 61, 291]

face_3d_model = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

prev_time = time.time()

# --- HAUPTSCHLEIFE ---
try:
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.flip(frame, 1)
        img_h, img_w, _ = frame.shape

        now = time.time()
        fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        # 1. YOLO Inferenz
        yolo_results = model(frame, verbose=False, conf=0.4, stream=True)
        detected_objects_list = []

        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                obj_name = model.names[cls]
                detected_objects_list.append(f"{obj_name}({conf:.2f})")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    f"{obj_name} {conf:.2f}",
                    (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2,
                )

        # 2. MediaPipe Face Mesh & DMS
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        avg_ear, mar = 0.0, 0.0
        blink_status, mouth_status = "Offen", "Normal"
        pose_state = "Geradeaus"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_ear = calculate_ear(LEFT_EYE, landmarks, img_w, img_h)
                right_ear = calculate_ear(RIGHT_EYE, landmarks, img_w, img_h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                    duration = time.time() - eyes_closed_start_time
                    blink_status = f"ZU ({duration:.1f}s)"
                else:
                    eyes_closed_start_time = None
                    blink_status = "Offen"

                mar = calculate_mar(MOUTH, landmarks, img_w, img_h)
                mouth_status = "Gaehnen" if mar > MAR_YAWN_THRESHOLD else "Normal"

                face_2d = []
                for idx in landmark_indices:
                    lm = landmarks[idx]
                    face_2d.append([int(lm.x * img_w), int(lm.y * img_h)])

                face_2d = np.array(face_2d, dtype=np.float64)
                cam_matrix = np.array(
                    [[img_w, 0, img_w / 2], [0, img_w, img_h / 2], [0, 0, 1]],
                    dtype=np.float64,
                )
                success_pnp, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d_model, face_2d, cam_matrix, np.zeros((4, 1))
                )

                if success_pnp:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles = cv2.RQDecomp3x3(rmat)[0]
                    raw_pitch = angles[0] * (-1)
                    raw_yaw = angles[1] * (-1)
                    raw_roll = angles[2]

                    smooth_yaw = alpha * raw_yaw + (1 - alpha) * smooth_yaw
                    smooth_pitch = alpha * raw_pitch + (1 - alpha) * smooth_pitch
                    smooth_roll = alpha * raw_roll + (1 - alpha) * smooth_roll

                    if smooth_yaw < -15:
                        pose_state = "Links"
                    elif smooth_yaw > 15:
                        pose_state = "Rechts"
                    elif smooth_pitch < -12:
                        pose_state = "Oben"
                    elif smooth_pitch > 12:
                        pose_state = "Unten"

        # 3. GPU Metriken auslesen (mit Fallbacks für RTX 3050)
        gpu_util, gpu_mem_used, power_watt = 0, 0.0, 0.0
        if nvml_active:
            try:
                gpu_util = nvmlDeviceGetUtilizationRates(handle).gpu
                mem = nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used = int(mem.used) / 1024**2
                power_watt = nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                pass

        # 4. HUD Rendering
        hud_height = 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, hud_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        hud_text_1 = (
            f"FPS: {fps:.1f} | GPU: {gpu_util}% | VRAM: {gpu_mem_used:.0f}MB | Power: {power_watt:.1f}W | Pose: {pose_state}"
        )
        hud_text_2 = (
            f"Augen: {blink_status} | Mund: {mouth_status} | Objekte: "
            f"{', '.join(detected_objects_list[:3]) if detected_objects_list else 'Keine'}"
        )

        cv2.putText(frame, hud_text_1, (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, hud_text_2, (12, 46), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. CSV Logging
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        csv_writer.writerow([
            timestamp_str,
            round(smooth_yaw, 2),
            round(smooth_pitch, 2),
            round(smooth_roll, 2),
            round(avg_ear, 2),
            round(mar, 2),
            pose_state,
            blink_status,
            mouth_status,
            str(detected_objects_list),
            gpu_util,
            round(gpu_mem_used, 1),
            round(power_watt, 1),
            round(fps, 1),
        ])

        cv2.imshow("Smart Cockpit - DMS & YOLO", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Sauberes Aufräumen & Analytics-Zusammenfassung
    cap.release()
    cv2.destroyAllWindows()
    if nvml_active:
        nvmlShutdown()
    csv_file.close()
    print(f"\n[INFO] Sitzung beendet. Log gespeichert unter: {csv_filename}")

    # Automatische Auswertung aus der CSV für den Performance-Nachweis
    try:
        df = pd.read_csv(csv_filename)
        if not df.empty:
            print("\n" + "="*40)
            print(" 📊 PERFORMANCE-ANALYTIK (RTX 3050)")
            print("="*40)
            print(f" • Durchschnittliche FPS:      {df['FPS'].mean():.2f}")
            print(f" • Durchschnittliche GPU-Last: {df['GPU_Util'].mean():.1f}%")
            print(f" • Durchschnittlicher VRAM:    {df['VRAM_MB'].mean():.1f} MB")
            print(f" • Durchschnittlicher Strom:   {df['Power_W'].mean():.2f} W")
            print("="*40)
    except Exception as e:
        print(f"Konnte CSV-Auswertung nicht erstellen: {e}")