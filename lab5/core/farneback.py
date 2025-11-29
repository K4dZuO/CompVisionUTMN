"""
Реализация алгоритма Farneback для вычисления плотного оптического потока.

Использует оптимизированную реализацию OpenCV для высокой производительности.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class FarnebackProcessor:
    """
    Процессор для вычисления плотного оптического потока методом Farneback.
    
    Алгоритм Farneback - это быстрый метод вычисления плотного оптического потока,
    оптимизированный в OpenCV. Работает быстрее, чем Horn-Schunck, но использует
    готовую реализацию OpenCV.
    
    Параметры:
    ----------
    pyr_scale : float
        Масштаб пирамиды (обычно 0.5)
    levels : int
        Количество уровней пирамиды
    winsize : int
        Размер окна усреднения
    iterations : int
        Количество итераций на каждом уровне
    poly_n : int
        Размер окна полинома (обычно 5 или 7)
    poly_sigma : float
        Стандартное отклонение Гаусса для сглаживания
    """
    
    def __init__(self,
                 pyr_scale: float = 0.5,
                 levels: int = 3,
                 winsize: int = 15,
                 iterations: int = 3,
                 poly_n: int = 5,
                 poly_sigma: float = 1.2):
        """
        Инициализация процессора Farneback.
        
        Args:
            pyr_scale: Масштаб пирамиды (0.5 = уменьшение в 2 раза)
            levels: Количество уровней пирамиды (больше = лучше для быстрых движений)
            winsize: Размер окна усреднения (больше = более гладкий поток)
            iterations: Количество итераций (больше = точнее, но медленнее)
            poly_n: Размер окна полинома (5 или 7)
            poly_sigma: Сигма Гаусса (больше = более гладкий поток)
        """
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление оптического потока между двумя кадрами.
        
        Args:
            frame1: Первый кадр (BGR или grayscale)
            frame2: Второй кадр (BGR или grayscale)
            
        Returns:
            Tuple (u, v):
            - u: Горизонтальная компонента потока
            - v: Вертикальная компонента потока
        """
        # Преобразование в grayscale если нужно
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1.copy()
            
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2.copy()
        
        # Вычисление плотного оптического потока методом Farneback
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,  # flow (None для первого вызова)
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            0  # flags
        )
        
        # Разделение на компоненты
        u = flow[..., 0]
        v = flow[..., 1]
        
        return u, v
    
    def compute_flow_magnitude_direction(self, 
                                        frame1: np.ndarray, 
                                        frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление оптического потока с величиной и направлением.
        
        Args:
            frame1: Первый кадр
            frame2: Второй кадр
            
        Returns:
            Tuple (u, v, magnitude, angle):
            - u: Горизонтальная компонента
            - v: Вертикальная компонента
            - magnitude: Величина потока
            - angle: Направление потока (в радианах)
        """
        u, v = self.compute_flow(frame1, frame2)
        
        # Вычисление величины и направления
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)
        
        return u, v, magnitude, angle
    
    def visualize_flow(self, 
                      frame: np.ndarray, 
                      u: np.ndarray, 
                      v: np.ndarray,
                      step: int = 16,
                      scale: float = 1.0) -> np.ndarray:
        """
        Визуализация оптического потока стрелками на сетке.
        
        Args:
            frame: Исходный кадр для отрисовки
            u: Горизонтальная компонента потока
            v: Вертикальная компонента потока
            step: Шаг сетки для отрисовки стрелок
            scale: Масштаб стрелок
            
        Returns:
            Изображение с визуализацией потока
        """
        vis_frame = frame.copy()
        h, w = u.shape
        
        # Создание сетки точек
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        # Получение векторов потока в точках сетки
        fx, fy = u[y, x], v[y, x]
        
        # Отрисовка стрелок
        for i in range(len(x)):
            pt1 = (x[i], y[i])
            pt2 = (int(x[i] + fx[i] * scale), int(y[i] + fy[i] * scale))
            
            # Цвет в зависимости от величины
            magnitude = np.sqrt(fx[i]**2 + fy[i]**2)
            if magnitude > 1.0:  # Рисуем только значимые векторы
                color = (0, 255, 0)  # Зеленый
                cv2.arrowedLine(vis_frame, pt1, pt2, color, 1, tipLength=0.3)
        
        return vis_frame
