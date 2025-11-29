"""
Модуль для визуализации оптического потока.

ВИЗУАЛИЗАЦИЯ ОПТИЧЕСКОГО ПОТОКА:
=================================

Разреженный поток (Sparse Flow):
---------------------------------
- Визуализация векторами-стрелками на ключевых точках
- Цветовое кодирование направления движения
- Длина стрелки пропорциональна скорости

Плотный поток (Dense Flow):
---------------------------
- HSV цветовое кодирование:
  * Hue (оттенок): направление движения (0-360°)
  * Saturation (насыщенность): величина потока (0-255)
  * Value (яркость): константа или дополнительная информация
- Heat maps: карты интенсивности движения
- Стрелки на регулярной сетке (опционально)

Оптимизации:
------------
- Эффективное использование NumPy для преобразований
- Минимизация копирований данных
- Кэширование цветовых карт
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class VisualizationEngine:
    """
    Движок для визуализации результатов оптического потока.
    
    Поддерживает различные режимы визуализации:
    - Разреженный поток (стрелки на ключевых точках)
    - Плотный поток (HSV цветовое кодирование)
    - Heat maps интенсивности
    - Комбинированная визуализация
    """
    
    def __init__(self):
        """Инициализация движка визуализации."""
        # Кэш для цветовой карты HSV
        self._hsv_colormap_cache: Optional[np.ndarray] = None
        
    def visualize_sparse_flow(self, frame: np.ndarray, points: np.ndarray,
                             vectors: np.ndarray, magnitudes: Optional[np.ndarray] = None,
                             scale: float = 1.0, thickness: int = 2,
                             color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Визуализация разреженного оптического потока стрелками.
        
        Args:
            frame: Исходный кадр
            points: Координаты точек (N, 2)
            vectors: Векторы потока (N, 2)
            magnitudes: Величины векторов (опционально, для цветового кодирования)
            scale: Масштаб стрелок
            thickness: Толщина линий
            color: Цвет стрелок (если None, используется цветовое кодирование по направлению)
            
        Returns:
            Изображение с визуализацией потока
        """
        vis_frame = frame.copy()
        
        if len(points) == 0:
            return vis_frame
        
        # Преобразование координат в целые числа
        points_int = points.astype(np.int32)
        vectors_scaled = vectors * scale
        
        if color is None:
            # Цветовое кодирование по направлению
            if magnitudes is None:
                magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
            
            # Нормализация величин для цветового кодирования
            if np.max(magnitudes) > 0:
                magnitudes_norm = magnitudes / np.max(magnitudes)
            else:
                magnitudes_norm = magnitudes
            
            # Вычисление направлений (углов)
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            
            # Рисование стрелок с цветовым кодированием
            for i in range(len(points_int)):
                pt1 = tuple(points_int[i])
                pt2 = (int(pt1[0] + vectors_scaled[i, 0]),
                      int(pt1[1] + vectors_scaled[i, 1]))
                
                # Цвет на основе направления и величины
                angle_deg = np.degrees(angles[i]) % 360
                hue = int(angle_deg / 360 * 179)  # OpenCV HSV: H в диапазоне [0, 179]
                saturation = int(magnitudes_norm[i] * 255)
                color_hsv = np.uint8([[[hue, saturation, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                color_tuple = tuple(map(int, color_bgr))
                
                # Рисование стрелки
                cv2.arrowedLine(vis_frame, pt1, pt2, color_tuple, thickness, 
                              tipLength=0.3)
        else:
            # Однотонные стрелки
            for i in range(len(points_int)):
                pt1 = tuple(points_int[i])
                pt2 = (int(pt1[0] + vectors_scaled[i, 0]),
                      int(pt1[1] + vectors_scaled[i, 1]))
                cv2.arrowedLine(vis_frame, pt1, pt2, color, thickness, tipLength=0.3)
        
        return vis_frame
    
    def visualize_dense_flow_hsv(self, u: np.ndarray, v: np.ndarray,
                                 magnitude_scale: float = 10.0) -> np.ndarray:
        """
        Визуализация плотного потока в формате HSV.
        
        HSV кодирование:
        - Hue (оттенок): направление движения (0-360°)
        - Saturation (насыщенность): нормализованная величина потока
        - Value (яркость): константа (максимальная яркость)
        
        Args:
            u: Горизонтальная компонента потока
            v: Вертикальная компонента потока
            magnitude_scale: Масштаб для нормализации величин
            
        Returns:
            Изображение в формате BGR для отображения
        """
        # Вычисление величины и направления
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)
        
        # Нормализация величины
        magnitude_norm = np.clip(magnitude / magnitude_scale, 0, 1)
        
        # Создание HSV изображения
        hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
        
        # Hue: направление (0-179 в OpenCV)
        hsv[..., 0] = (np.degrees(angle) + 180) % 360 / 2
        
        # Saturation: величина потока
        hsv[..., 1] = (magnitude_norm * 255).astype(np.uint8)
        
        # Value: максимальная яркость
        hsv[..., 2] = 255
        
        # Преобразование в BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return bgr
    
    def visualize_heatmap(self, magnitude: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Визуализация величины потока в виде heat map.
        
        Args:
            magnitude: Матрица величин потока
            colormap: Цветовая карта OpenCV
            
        Returns:
            Изображение heat map в формате BGR
        """
        # Нормализация для визуализации
        magnitude_norm = magnitude.astype(np.float32)
        if np.max(magnitude_norm) > 0:
            magnitude_norm = magnitude_norm / np.max(magnitude_norm) * 255
        else:
            magnitude_norm = magnitude_norm * 255
        
        magnitude_uint8 = np.clip(magnitude_norm, 0, 255).astype(np.uint8)
        
        # Применение цветовой карты
        heatmap = cv2.applyColorMap(magnitude_uint8, colormap)
        
        return heatmap
    
    def visualize_flow_grid(self, frame: np.ndarray, u: np.ndarray, v: np.ndarray,
                           step: int = 20, scale: float = 1.0,
                           thickness: int = 1) -> np.ndarray:
        """
        Визуализация плотного потока стрелками на регулярной сетке.
        
        Args:
            frame: Исходный кадр
            u: Горизонтальная компонента потока
            v: Вертикальная компонента потока
            step: Шаг сетки (пиксели)
            scale: Масштаб стрелок
            thickness: Толщина линий
            
        Returns:
            Изображение с визуализацией
        """
        vis_frame = frame.copy()
        height, width = u.shape
        
        # Создание сетки точек
        y, x = np.mgrid[step//2:height:step, step//2:width:step]
        
        # Извлечение векторов потока в точках сетки
        flow_x = u[y, x]
        flow_y = v[y, x]
        
        # Рисование стрелок
        for i in range(len(y)):
            for j in range(len(x[0])):
                pt1 = (int(x[i, j]), int(y[i, j]))
                pt2 = (int(pt1[0] + flow_x[i, j] * scale),
                      int(pt1[1] + flow_y[i, j] * scale))
                
                # Вычисление цвета на основе направления
                angle = np.arctan2(flow_y[i, j], flow_x[i, j])
                magnitude = np.sqrt(flow_x[i, j]**2 + flow_y[i, j]**2)
                
                angle_deg = np.degrees(angle) % 360
                hue = int(angle_deg / 360 * 179)
                saturation = int(np.clip(magnitude * 10, 0, 255))
                color_hsv = np.uint8([[[hue, saturation, 255]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                color_tuple = tuple(map(int, color_bgr))
                
                cv2.arrowedLine(vis_frame, pt1, pt2, color_tuple, thickness, tipLength=0.3)
        
        return vis_frame
    
    def visualize_tracked_objects(self, frame: np.ndarray, 
                                tracked_objects: dict, 
                                paths: Optional[dict] = None,
                                mask: Optional[np.ndarray] = None,
                                u: Optional[np.ndarray] = None,
                                v: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Визуализация отслеживаемых объектов с плотным потоком штрихами.
        
        Args:
            frame: Исходный кадр
            tracked_objects: Словарь ID -> (x, y, w, h)
            paths: Словарь ID -> Trajectory (deque of points)
            mask: Маска движения (опционально, для наложения)
            u, v: Компоненты оптического потока (для штрихов)
            
        Returns:
            Изображение с визуализацией
        """
        vis_frame = frame.copy()
        
        # 1. Рисование штрихов плотного потока (если есть u, v)
        if u is not None and v is not None:
            # Рисуем штрихи только там, где есть движение (mask) или везде
            # Для "плотного потока штрихами" обычно рисуют сетку
            step = 20
            scale = 1.0
            height, width = u.shape
            y, x = np.mgrid[step//2:height:step, step//2:width:step]
            
            flow_x = u[y, x]
            flow_y = v[y, x]
            
            # Фильтрация по маске (если есть) или по величине
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            mask_indices = magnitude > 1.0  # Рисуем только если есть движение
            
            y = y[mask_indices]
            x = x[mask_indices]
            flow_x = flow_x[mask_indices]
            flow_y = flow_y[mask_indices]
            
            for i in range(len(y)):
                pt1 = (int(x[i]), int(y[i]))
                pt2 = (int(pt1[0] + flow_x[i] * scale),
                      int(pt1[1] + flow_y[i] * scale))
                
                # Цвет штриха (зеленый или по направлению)
                color = (0, 255, 0) 
                cv2.line(vis_frame, pt1, pt2, color, 1)
                cv2.circle(vis_frame, pt1, 1, color, -1)

        # 2. Если есть маска, накладываем её полупрозрачно (опционально, можно убрать если мешает штрихам)
        # if mask is not None:
        #     mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #     vis_frame = cv2.addWeighted(vis_frame, 0.8, mask_bgr, 0.2, 0)
            
        # 3. Рисование bounding box'ов и ID
        for object_id, (x, y, w, h) in tracked_objects.items():
            # Генерация цвета на основе ID
            np.random.seed(object_id)
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            
            # Bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # ID
            text = f"ID {object_id}"
            cv2.putText(vis_frame, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Центроид
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            cv2.circle(vis_frame, (cX, cY), 4, color, -1)
            
            # Траектория (Strokes)
            if paths is not None and object_id in paths:
                pts = paths[object_id]
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(vis_frame, pts[i - 1], pts[i], color, 2)
            
        return vis_frame
