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
    
    def visualize_dense_flow_overlay(self, frame: np.ndarray, u: np.ndarray, v: np.ndarray,
                                    alpha: float = 0.7, magnitude_scale: float = 10.0) -> np.ndarray:
        """
        Наложение плотного потока на исходный кадр.
        
        Args:
            frame: Исходный кадр
            u: Горизонтальная компонента потока
            v: Вертикальная компонента потока
            alpha: Прозрачность наложения (0-1)
            magnitude_scale: Масштаб для нормализации величин
            
        Returns:
            Комбинированное изображение
        """
        # Получение HSV визуализации
        flow_vis = self.visualize_dense_flow_hsv(u, v, magnitude_scale)
        
        # Наложение с прозрачностью
        result = cv2.addWeighted(frame, 1 - alpha, flow_vis, alpha, 0)
        
        return result
    
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
    
    def create_legend(self, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """
        Создание легенды для визуализации потока.
        
        Показывает соответствие цветов направлениям движения.
        
        Args:
            size: Размер легенды (width, height)
            
        Returns:
            Изображение легенды
        """
        width, height = size
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Создание радиального градиента для направлений
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2 - 10
        
        for y in range(height):
            for x in range(width):
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist <= radius:
                    angle = np.arctan2(dy, dx)
                    angle_deg = np.degrees(angle) % 360
                    
                    # Нормализация расстояния для насыщенности
                    sat = int((1 - dist / radius) * 255)
                    
                    hue = int(angle_deg / 360 * 179)
                    color_hsv = np.uint8([[[hue, sat, 255]]])
                    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                    legend[y, x] = color_bgr
        
        # Добавление стрелок направлений
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        for angle_deg in directions:
            angle_rad = np.radians(angle_deg)
            end_x = int(center_x + radius * 0.8 * np.cos(angle_rad))
            end_y = int(center_y + radius * 0.8 * np.sin(angle_rad))
            cv2.arrowedLine(legend, (center_x, center_y), (end_x, end_y),
                          (255, 255, 255), 2, tipLength=0.2)
        
        return legend

