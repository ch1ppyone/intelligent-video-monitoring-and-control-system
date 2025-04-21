import os
import cv2
import argparse
import random
import numpy as np
import tensorflow as tf
import ssl
import sqlite3

ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import mediapipe as mp
from sklearn.utils import class_weight

def random_augment(frame):
    if np.random.rand() < 0.5:
        frame = cv2.flip(frame, 1)
    factor = 0.8 + np.random.rand() * 0.4
    frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
    return frame

class VideoDataGenerator(Sequence):
    def __init__(self, directory, img_size=224, batch_size=32,
                 subset='training', validation_split=0.2, augment=False, shuffle=True, seed=42):
        super().__init__()
        self.directory = directory
        self.img_size = img_size
        self.batch_size = batch_size
        self.subset = subset
        self.validation_split = validation_split
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.samples = []
        self.class_indices = {}
        self._scan_directory()
        self.on_epoch_end()
        self.num_keypoints = 33 * 3

    def _scan_directory(self):
        # Получаем классы из базы данных
        with sqlite3.connect("states.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT code FROM states")
            classes = [row[0] for row in cursor.fetchall()]
        self.class_indices = {cls_name: idx for idx, cls_name in enumerate(classes)}
        all_samples = []
        for cls_name in classes:
            cls_dir = os.path.join(self.directory, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith('.mp4'):
                    video_path = os.path.join(cls_dir, fname)
                    all_samples.append((video_path, self.class_indices[cls_name]))
        random.seed(self.seed)
        random.shuffle(all_samples)
        total = len(all_samples)
        split_idx = int(total * (1 - self.validation_split))
        if self.subset == 'training':
            self.samples = all_samples[:split_idx]
        elif self.subset == 'validation':
            self.samples = all_samples[split_idx:]
        else:
            raise ValueError("subset должно быть 'training' или 'validation'")
        self.num_classes = len(classes)
        print(f"Найдено {len(all_samples)} видео, из них для {self.subset}: {len(self.samples)}. Классов: {self.num_classes}")

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(self.batch_size)))

    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x_images = []
        batch_x_keypoints = []
        batch_y = []
        with mp.solutions.pose.Pose(static_image_mode=True) as pose:
            for video_path, label in batch_samples:
                frame = self._get_random_frame(video_path)
                if frame is None:
                    continue
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                if self.augment:
                    frame = random_augment(frame)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_norm = image_rgb.astype(np.float32) / 255.0
                result = pose.process(image_rgb)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                else:
                    keypoints = np.zeros(self.num_keypoints, dtype=np.float32)
                batch_x_images.append(image_norm)
                batch_x_keypoints.append(keypoints)
                batch_y.append(label)
        if len(batch_x_images) == 0:
            return ((np.empty((0, self.img_size, self.img_size, 3)),
                     np.empty((0, self.num_keypoints))),
                    np.empty((0, self.num_classes)))
        batch_x_images = np.array(batch_x_images, dtype=np.float32)
        batch_x_keypoints = np.array(batch_x_keypoints, dtype=np.float32)
        batch_y = to_categorical(np.array(batch_y), num_classes=self.num_classes)
        return ((batch_x_images, batch_x_keypoints), batch_y)

    def _get_random_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Не удалось открыть видео: {video_path}")
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None
        frame_idx = random.randint(0, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Не удалось считать кадр {frame_idx} из видео: {video_path}")
            return None
        return frame

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)

def main():
    parser = argparse.ArgumentParser(description="Дообучение модели на видео (mp4), организованных по папкам с кодами состояний.")
    parser.add_argument('--train_dir', type=str, default="train", help="Путь к папке с данными (по умолчанию: train)")
    parser.add_argument('--img_size', type=int, default=224, help="Размер стороны изображения (по умолчанию: 224)")
    parser.add_argument('--batch_size', type=int, default=32, help="Размер батча (по умолчанию: 32)")
    parser.add_argument('--epochs', type=int, default=10, help="Количество эпох (по умолчанию: 10)")
    parser.add_argument('--model_out', type=str, default="model.keras", help="Путь для сохранения дообученной модели (по умолчанию: model.keras)")
    args = parser.parse_args()
    train_generator = VideoDataGenerator(directory=args.train_dir, img_size=args.img_size, batch_size=args.batch_size, subset='training', validation_split=0.2, augment=True, shuffle=True)
    validation_generator = VideoDataGenerator(directory=args.train_dir, img_size=args.img_size, batch_size=args.batch_size, subset='validation', validation_split=0.2, augment=False, shuffle=False)
    if len(train_generator.samples) == 0:
        print(f"В папке '{args.train_dir}' не найдено ни одного видео. Проверьте структуру папок и форматы файлов.")
        return
    y_train = [label for _, label in train_generator.samples]
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {int(cls): weight for cls, weight in zip(classes, weights)}
    print("Веса классов:", class_weights)
    image_input = Input(shape=(args.img_size, args.img_size, 3), name='image_input')
    keypoints_input = Input(shape=(33*3,), name='keypoints_input')
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(args.img_size, args.img_size, 3))
    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(64, activation='relu')(keypoints_input)
    y = Dropout(0.5)(y)
    y = Dense(32, activation='relu')(y)
    combined = concatenate([x, y])
    output = Dense(train_generator.num_classes, activation='softmax')(combined)
    model = Model(inputs=[image_input, keypoints_input], outputs=output)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(args.model_out, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    print("Этап 1: Обучение только добавленных слоёв")
    model.fit(train_generator, validation_data=validation_generator, epochs=args.epochs, callbacks=[checkpoint, earlystop, reduce_lr], class_weight=class_weights)
    N = 50
    for layer in base_model.layers[-N:]:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Этап 2: Fine-tuning модели")
    model.fit(train_generator, validation_data=validation_generator, epochs=5, callbacks=[checkpoint, earlystop, reduce_lr], class_weight=class_weights)
    print(f"Обучение завершено. Лучшая модель сохранена в '{args.model_out}'.")

if __name__ == "__main__":
    main()