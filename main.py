import os
import cv2
import numpy as np
import pandas as pd
import time
import datetime
import base64
import random
import threading
import warnings
from queue import Queue, Full, Empty
import sqlite3
import subprocess
import shlex

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
import tensorflow as tf

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

app = Flask(__name__, template_folder="template")
socketio = SocketIO(app, cors_allowed_origins="*")

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU настроен с динамическим ростом памяти.")


def get_color_for_track(track_id):
    random.seed(track_id)
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    return f"rgb({r},{g},{b})"


class DatabaseManager:
    def __init__(self, db_path="states.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS states (
                    code TEXT PRIMARY KEY,
                    description TEXT NOT NULL
                )
            """)
            cursor.execute("SELECT COUNT(*) FROM states")
            count = cursor.fetchone()[0]
            if count == 0:
                initial_states = [
                    ("S0", "Стоит, не двигается"),
                    ("S1", "Двигается и работает"),
                    ("Sч1_0001", "Сортировка"),
                    ("S1_0002", "Злость"),
                    ("S1_0003", "Нажатие кнопки"),
                    ("S1_0004", "Поиск"),
                    ("S1_0005", "Бег"),
                    ("S1_0006", "Ходьба"),
                    ("S1_0007", "Поднятие предмета"),
                    ("S1_0008", "Транспортировка"),
                ]
                cursor.executemany("INSERT INTO states (code, description) VALUES (?, ?)", initial_states)
            conn.commit()

    def get_states(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT code, description FROM states")
            return {row[0]: row[1] for row in cursor.fetchall()}

    def add_state(self, code, description):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO states (code, description) VALUES (?, ?)", (code, description))
                conn.commit()
                os.makedirs(os.path.join("train", code), exist_ok=True)
                return True
            except sqlite3.IntegrityError:
                return False

    def update_state(self, old_code, new_code, description):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("UPDATE states SET code = ?, description = ? WHERE code = ?",
                               (new_code, description, old_code))
                if cursor.rowcount > 0:
                    conn.commit()
                    if old_code != new_code:
                        os.rename(os.path.join("train", old_code), os.path.join("train", new_code))
                    return True
                return False
            except sqlite3.IntegrityError:
                return False

    def delete_state(self, code):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM states WHERE code = ?", (code,))
            if cursor.rowcount > 0:
                conn.commit()
                state_folder = os.path.join("train", code)
                if os.path.exists(state_folder):
                    for f in os.listdir(state_folder):
                        os.remove(os.path.join(state_folder, f))
                    os.rmdir(state_folder)
                return True
            return False


class VideoProcessor:
    def __init__(self, source="test_data/default.mp4"):
        self.source = source
        self.cap = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise Exception(f"Не удалось открыть видео источник: {source}")
        if not source.isdigit() and source.startswith("rtsp://"):
            time.sleep(1)
        self.lock = threading.Lock()
        self.frame_queue = Queue(maxsize=10)
        threading.Thread(target=self._read_frames, daemon=True).start()
        self.detector = YOLO("yolo12x.pt")
        self.deepsort = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=0.6,
            max_iou_distance=0.5,
            max_cosine_distance=0.5,
            embedder="mobilenet"
        )
        self.model = None
        if os.path.exists("model.keras"):
            try:
                self.model = tf.keras.models.load_model("model.keras")
                print("Модель успешно загружена из model.keras")
            except Exception as e:
                print("Ошибка загрузки модели:", e)
        else:
            print("model.keras не найден, fallback отключён")
        self.db_manager = DatabaseManager()
        self.states_mapping = self.db_manager.get_states()
        self.state_counts = {}
        self.log_messages = []

    def _read_frames(self):
        while True:
            with self.lock:
                ret, frame = self.cap.read()
            if not ret or frame is None:
                if os.path.exists(self.source):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(frame, block=False)
            except Full:
                pass
            time.sleep(0.01)

    def process_frame(self, frame, pose_estimator):
        if frame is None:
            return None
        results = self.detector.predict(source=frame, conf=0.4, classes=0, imgsz=640, verbose=False)
        boxes = results[0].boxes
        bboxes_xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
        for i, box in enumerate(bboxes_xyxy):
            x1, y1, x2, y2 = box[:4].astype(int)
            if (x2 - x1 < 1) or (y2 - y1 < 1):
                continue
            if y2 > frame.shape[0] or x2 > frame.shape[1]:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            track_id = i
            crop = frame[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_result = pose_estimator.process(rgb_crop)
            if self.model is not None:
                try:
                    img_input = cv2.resize(crop, (224, 224))
                    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                    img_input = img_input.astype(np.float32) / 255.0
                    img_input = np.expand_dims(img_input, axis=0)
                    if pose_result.pose_landmarks:
                        landmarks = pose_result.pose_landmarks.landmark
                        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                    else:
                        keypoints = np.zeros(33 * 3, dtype=np.float32)
                    keypoints = np.expand_dims(keypoints, axis=0)
                    pred = self.model.predict([img_input, keypoints])
                    class_index = int(np.argmax(pred))
                    labels = list(self.states_mapping.keys())
                    raw_state = labels[class_index] if class_index < len(labels) else f"S{class_index}"
                except Exception as e:
                    print(f"Ошибка при классификации для track {track_id}: {e}")
                    raw_state = "Err"
            else:
                raw_state = "Err"
            log_state = self.states_mapping.get(raw_state, raw_state)
            self.state_counts.setdefault(track_id, {})
            self.state_counts[track_id][raw_state] = self.state_counts[track_id].get(raw_state, 0) + 1
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"{timestamp} - Track {track_id}: {log_state}"
            self.log_messages.append((track_id, message))
            if pose_result.pose_landmarks:
                w_box = x2 - x1
                h_box = y2 - y1
                landmarks = pose_result.pose_landmarks.landmark
                for connection in mp.solutions.pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]
                    x_start = x1 + int(start.x * w_box)
                    y_start = y1 + int(start.y * h_box)
                    x_end = x1 + int(end.x * w_box)
                    y_end = y1 + int(end.y * h_box)
                    cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), 2)
                for lm in landmarks:
                    x_lm = x1 + int(lm.x * w_box)
                    y_lm = y1 + int(lm.y * h_box)
                    cv2.circle(frame, (x_lm, y_lm), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"ID{track_id} {raw_state}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)
        return frame

    def get_processed_frame_with_pose(self, pose_estimator):
        try:
            frame = self.frame_queue.get(timeout=1)
        except Empty:
            return None
        processed_frame = self.process_frame(frame, pose_estimator)
        return processed_frame

    def release(self):
        self.cap.release()


video_processor = VideoProcessor("test_data/default.mp4")


@app.route("/")
def index():
    test_folder = "test_data"
    test_videos = []
    if os.path.exists(test_folder):
        for f in os.listdir(test_folder):
            if f.lower().endswith(".mp4"):
                test_videos.append(os.path.join(test_folder, f))
    return render_template("index.html", test_videos=test_videos, current_source=video_processor.source)


@app.route("/set_video_source", methods=["POST"])
def set_video_source():
    new_source = request.form.get("video_source")
    if new_source:
        if new_source == "webcam":
            new_source = "0"
        elif new_source == "rtsp":
            rtsp_url = request.form.get("rtsp_url")
            if rtsp_url:
                new_source = rtsp_url
            else:
                return jsonify({"error": "RTSP URL не задан"}), 400
        global video_processor
        video_processor.release()
        try:
            video_processor = VideoProcessor(new_source)
            return jsonify({"message": "Видео источник изменён"}), 200
        except Exception as e:
            return jsonify({"error": f"Ошибка открытия источника: {e}"}), 400
    return jsonify({"error": "Ошибка"}), 400


@app.route("/get_states", methods=["GET"])
def get_states():
    states = video_processor.db_manager.get_states()
    return jsonify({"states": [{"code": k, "description": v} for k, v in states.items()]})


@app.route("/add_state", methods=["POST"])
def add_state():
    data = request.get_json()
    code = data.get("code")
    description = data.get("description")
    if not code or not description:
        return jsonify({"error": "Код и описание обязательны"}), 400
    if video_processor.db_manager.add_state(code, description):
        video_processor.states_mapping = video_processor.db_manager.get_states()
        return jsonify({"message": "Состояние добавлено"}), 200
    return jsonify({"error": "Состояние с таким кодом уже существует"}), 400


@app.route("/update_state", methods=["POST"])
def update_state():
    data = request.get_json()
    old_code = data.get("old_code")
    new_code = data.get("new_code")
    description = data.get("description")
    if not old_code or not new_code or not description:
        return jsonify({"error": "Все поля обязательны"}), 400
    if video_processor.db_manager.update_state(old_code, new_code, description):
        video_processor.states_mapping = video_processor.db_manager.get_states()
        return jsonify({"message": "Состояние обновлено"}), 200
    return jsonify({"error": "Не удалось обновить состояние"}), 400


@app.route("/delete_state", methods=["POST"])
def delete_state():
    data = request.get_json()
    code = data.get("code")
    if not code:
        return jsonify({"error": "Код состояния обязателен"}), 400
    if video_processor.db_manager.delete_state(code):
        video_processor.states_mapping = video_processor.db_manager.get_states()
        return jsonify({"message": "Состояние удалено"}), 200
    return jsonify({"error": "Не удалось удалить состояние"}), 400


@app.route("/get_videos/<state_code>", methods=["GET"])
def get_videos(state_code):
    state_folder = os.path.join("train", state_code)
    videos = []
    if os.path.exists(state_folder):
        videos = [f for f in os.listdir(state_folder) if f.lower().endswith(".mp4")]
    return jsonify({"videos": videos})


@app.route("/upload_video/<state_code>", methods=["POST"])
def upload_video(state_code):
    if "video" not in request.files:
        return jsonify({"error": "Видео не выбрано"}), 400
    file = request.files["video"]
    if not file.filename.lower().endswith(".mp4"):
        return jsonify({"error": "Только MP4 файлы разрешены"}), 400
    state_folder = os.path.join("train", state_code)
    os.makedirs(state_folder, exist_ok=True)
    video_path = os.path.join(state_folder, file.filename)
    if os.path.exists(video_path):
        return jsonify({"error": "Видео с таким именем уже существует"}), 400
    file.save(video_path)
    return jsonify({"message": "Видео загружено"}), 200


@app.route("/delete_video/<state_code>", methods=["POST"])
def delete_video(state_code):
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "Имя файла обязательно"}), 400
    video_path = os.path.join("train", state_code, filename)
    if os.path.exists(video_path):
        os.remove(video_path)
        return jsonify({"message": "Видео удалено"}), 200
    return jsonify({"error": "Видео не найдено"}), 400


@app.route("/train_model", methods=["POST"])
def train_model():
    def run_training():
        # Проверка наличия видео для всех состояний
        states = video_processor.db_manager.get_states()
        missing_states = []
        for state_code in states.keys():
            state_folder = os.path.join("train", state_code)
            if not os.path.exists(state_folder) or not any(
                    f.lower().endswith(".mp4") for f in os.listdir(state_folder)):
                missing_states.append(state_code)
        if missing_states:
            socketio.emit('training_warning', {
                'message': f'Отсутствуют видео для состояний: {", ".join(missing_states)}. Обучение может быть неполным.'})
            return

        cmd = "python train.py --train_dir train"
        process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                socketio.emit('training_log', {'log': output.strip()})
        return_code = process.poll()
        if return_code == 0:
            socketio.emit('training_complete', {'message': 'Обучение завершено успешно'})
        else:
            socketio.emit('training_complete', {'message': 'Ошибка при обучении'})

    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({"message": "Обучение начато"}), 200


def frame_emit_thread():
    with mp.solutions.pose.Pose(static_image_mode=True) as pose_estimator:
        while True:
            frame = video_processor.get_processed_frame_with_pose(pose_estimator)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    b64_frame = base64.b64encode(buffer).decode('utf-8')
                    socketio.emit('video_frame', {'data': b64_frame})
            socketio.sleep(0.03)


def data_emit_thread():
    while True:
        logs = video_processor.log_messages[-100:]
        socketio.emit('log_update', {'logs': logs})
        socketio.emit('chart_update', {'state_counts': video_processor.state_counts})
        socketio.sleep(2)


@socketio.on('connect')
def handle_connect():
    print("Клиент подключён")
    socketio.start_background_task(frame_emit_thread)
    socketio.start_background_task(data_emit_thread)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 9000
    print(f"Сервер запущен по адресу: http://localhost:{port}")
    socketio.run(app, host=host, port=port, debug=True, allow_unsafe_werkzeug=True)