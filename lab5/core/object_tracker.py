"""
Реализация отслеживания объектов (Object Tracking).

Использует метод вычитания фона (Background Subtraction) для детектирования
движущихся объектов и алгоритм центроидного отслеживания (Centroid Tracking)
для присвоения и сохранения уникальных ID объектов между кадрами.
"""

import numpy as np
import cv2
import copy
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict, deque
from core.horn_schunck import HornSchunckProcessor

class CentroidTracker:
    """
    Простой трекер, основанный на центроидах.
    Сопоставляет новые обнаруженные объекты с существующими на основе Евклидова расстояния.
    """
    def __init__(self, max_disappeared: int = 50, max_distance: int = 50, max_trace: int = 30):
        """
        Args:
            max_disappeared: Максимальное количество кадров, которое объект может отсутствовать.
            max_distance: Максимальное расстояние (в пикселях) между центроидами для сопоставления.
            max_trace: Максимальная длина хвоста (траектории) объекта.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # ID -> centroid (x, y)
        self.disappeared = OrderedDict() # ID -> frames_disappeared
        self.bboxes = OrderedDict() # ID -> (x, y, w, h)
        self.paths = OrderedDict() # ID -> deque of (x, y)
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_trace = max_trace

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int]):
        """Регистрация нового объекта."""
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.paths[self.next_object_id] = deque(maxlen=self.max_trace)
        self.paths[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id: int):
        """Удаление объекта из отслеживания."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]
        del self.paths[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Tuple[Dict, Dict]:
        """
        Обновление состояния трекера.
        
        Returns:
            Tuple (objects, paths):
            - objects: Словарь ID -> Bounding Box
            - paths: Словарь ID -> Trajectory (deque of points)
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return dict(self.bboxes), copy.deepcopy(self.paths)

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(tuple(input_centroids[i]), rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = []
            for oc in object_centroids:
                row = []
                for ic in input_centroids:
                    dist = np.linalg.norm(np.array(oc) - np.array(ic))
                    row.append(dist)
                D.append(row)
            D = np.array(D)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                centroid = tuple(input_centroids[col])
                
                self.objects[object_id] = centroid
                self.bboxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                self.paths[object_id].append(centroid)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(tuple(input_centroids[col]), rects[col])

        return dict(self.bboxes), copy.deepcopy(self.paths)


class ObjectTracker:
    """
    Класс для детектирования и отслеживания движущихся объектов.
    """
    def __init__(self, min_area: int = 500, max_disappeared: int = 40, max_distance: int = 50):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.tracker = CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
        self.min_area = min_area
        # Используем наш класс HornSchunckProcessor для вычисления потока
        # Меньше итераций для скорости в реальном времени
        self.flow_processor = HornSchunckProcessor(lambda_val=1.0, num_iterations=30)
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Обработка кадра (BG Subtraction).
        """
        mask = self.bg_subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            (x, y, w, h) = cv2.boundingRect(contour)
            rects.append((x, y, w, h))
            
        objects, paths = self.tracker.update(rects)
        
        return objects, paths, mask

    def update_from_mask(self, mask: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Обновление трекера на основе бинарной маски движения.
        """
        # Улучшенная морфология для объединения частей объектов
        # 1. Закрытие (closing) - заполняет дыры внутри объектов
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. Открытие (opening) - убирает мелкий шум
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # 3. Дилатация - расширяет объекты для лучшего объединения
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate, iterations=2)
        
        # Поиск контуров
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            (x, y, w, h) = cv2.boundingRect(contour)
            rects.append((x, y, w, h))
            
        objects, paths = self.tracker.update(rects)
        
        return objects, paths
