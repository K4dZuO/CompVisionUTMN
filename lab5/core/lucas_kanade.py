"""
Реализация алгоритма Лукаса-Канаде для вычисления разреженного оптического потока.

АЛГОРИТМ ЛУКАСА-КАНАДЕ:
======================

Математическая основа:
----------------------
Алгоритм основан на предположении локального постоянства потока:
в небольшой окрестности точки все пиксели движутся с одинаковой скоростью.

Для каждой точки (x, y) с окрестностью Ω (например, окно 5x5):
- Все точки в окрестности имеют одинаковый вектор потока (u, v)
- Brightness constancy: I(x, y, t) = I(x + u, y + v, t + 1)

Система уравнений:
------------------
Для окрестности Ω размера (2k+1) x (2k+1) получаем систему из (2k+1)² уравнений:
I_x(x_i, y_i)*u + I_y(x_i, y_i)*v + I_t(x_i, y_i) = 0  для всех (x_i, y_i) ∈ Ω

В матричном виде:
[A] [u]   [b]
    [v] = 

где:
A = [I_x(x_1, y_1)  I_y(x_1, y_1)]
    [I_x(x_2, y_2)  I_y(x_2, y_2)]
    [...            ...           ]

b = [-I_t(x_1, y_1)]
    [-I_t(x_2, y_2)]
    [...           ]

Решение методом наименьших квадратов:
--------------------------------------
[u]   [ΣI_x²   ΣI_xI_y]⁻¹ [ΣI_xI_t]
[v] = [ΣI_xI_y ΣI_y²  ]   [ΣI_yI_t]

где суммы берутся по всем точкам в окрестности Ω.

Условие разрешимости:
---------------------
Матрица [ΣI_x²   ΣI_xI_y] должна быть обратимой (не вырожденной).
         [ΣI_xI_y ΣI_y²  ]

Это означает, что в окрестности должны присутствовать градиенты в обоих
направлениях (структурированная текстура, углы). Плоские или одномерные
области (aperture problem) не дают надежного решения.

Пирамидальный подход:
---------------------
Для обработки больших перемещений используется пирамида изображений:
1. Построение гауссовой пирамиды (несколько уровней с уменьшением разрешения)
2. Вычисление потока на верхнем уровне (низкое разрешение, большие перемещения)
3. Уточнение потока на следующих уровнях вниз
4. Финальное уточнение на исходном разрешении

Преимущества:
-------------
- Быстрое вычисление (только для ключевых точек)
- Устойчивость к шуму благодаря усреднению по окрестности
- Эффективен для структурированных текстур

Недостатки:
-----------
- Разреженный поток (только для "хороших" точек)
- Не работает для плоских областей
- Ограничен размером окна (не может отследить большие перемещения без пирамиды)

Детектирование ключевых точек:
-------------------------------
Используются методы Shi-Tomasi или Harris для определения "хороших для отслеживания" точек:
- Углы (corners)
- Точки с градиентами в обоих направлениях
- Точки с достаточной вариативностью текстуры
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class LucasKanadeProcessor:
    """
    Процессор для вычисления оптического потока методом Лукаса-Канаде.
    
    Оптимизации:
    - Пирамидальное вычисление для больших перемещений
    - Эффективное детектирование ключевых точек
    - Векторизованные вычисления производных
    - Кэширование структур данных
    """
    
    def __init__(self, window_size: int = 15, max_level: int = 2, 
                 max_corners: int = 500, quality_level: float = 0.01,
                 min_distance: int = 10):
        """
        Инициализация процессора Лукаса-Канаде.
        
        Args:
            window_size: Размер окна для вычисления потока (2k+1)
            max_level: Количество уровней пирамиды
            max_corners: Максимальное количество отслеживаемых точек
            quality_level: Порог качества для детектирования углов
            min_distance: Минимальное расстояние между точками
        """
        self.window_size = window_size
        self.max_level = max_level
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        
        # Параметры для алгоритма отслеживания OpenCV
        self.lk_params = dict(
            winSize=(window_size, window_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Параметры для детектирования углов
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=3,
            useHarrisDetector=False,  # Используем Shi-Tomasi
            k=0.04
        )
        
        # Кэш для предыдущих точек
        self._prev_points = None
        self._prev_frame = None
        
    def detect_features(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Детектирование ключевых точек для отслеживания.
        
        Использует метод Shi-Tomasi (Good Features to Track) для нахождения
        точек с хорошими характеристиками для отслеживания:
        - Углы
        - Точки с достаточной текстурной информацией
        - Точки, которые можно однозначно локализовать
        
        Args:
            frame: Кадр изображения (градации серого)
            mask: Маска области интереса (опционально)
            
        Returns:
            Массив точек формы (N, 1, 2) для использования с OpenCV
        """
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детектирование углов методом Shi-Tomasi
        corners = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)
        
        if corners is None:
            return np.array([], dtype=np.float32).reshape(0, 1, 2)
        
        return corners
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray, 
                    prev_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление оптического потока методом Лукаса-Канаде.
        
        Использует пирамидальную версию алгоритма (Lucas-Kanade pyramid)
        для обработки больших перемещений.
        
        Args:
            frame1: Первый кадр
            frame2: Второй кадр
            prev_points: Точки из предыдущего кадра (если None, будут детектированы)
            
        Returns:
            Tuple (next_points, status, error):
            - next_points: Новые позиции точек
            - status: Статус отслеживания (1 = успешно, 0 = потеряно)
            - error: Ошибка отслеживания
        """
        # Преобразование в градации серого
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1.copy()
            
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2.copy()
        
        # Детектирование точек если не предоставлены
        if prev_points is None:
            prev_points = self.detect_features(gray1)
            if prev_points.shape[0] == 0:
                # Нет точек для отслеживания
                return (np.array([]).reshape(0, 1, 2), 
                       np.array([], dtype=np.uint8), 
                       np.array([], dtype=np.float32))
        
        # Вычисление оптического потока пирамидальным методом (OpenCV)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, prev_points, None, **self.lk_params
        )
        
        return next_points, status, error
    
    def compute_flow_manual(self, frame1: np.ndarray, frame2: np.ndarray,
                           prev_points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        РУЧНАЯ РЕАЛИЗАЦИЯ алгоритма Лукаса-Канаде.
        
        Реализует классический алгоритм Лукаса-Канаде методом наименьших квадратов:
        Для каждой точки с окрестностью Ω размера (2k+1) x (2k+1):
        - Вычисляет производные I_x, I_y, I_t в окрестности
        - Решает систему уравнений методом наименьших квадратов:
          [ΣI_x²   ΣI_xI_y] [u]   [ΣI_xI_t]
          [ΣI_xI_y ΣI_y²  ] [v] = [ΣI_yI_t]
        
        Математическая основа:
        ----------------------
        Предположение локального постоянства потока: все пиксели в окрестности
        движутся с одинаковой скоростью (u, v).
        
        Brightness constancy: I(x, y, t) = I(x + u, y + v, t + 1)
        Линеаризация: I_x*u + I_y*v + I_t = 0
        
        Для окрестности получаем систему из (2k+1)² уравнений, которая решается
        методом наименьших квадратов.
        
        Args:
            frame1: Первый кадр
            frame2: Второй кадр
            prev_points: Точки для отслеживания (если None, будут детектированы)
            
        Returns:
            Tuple (next_points, status, error):
            - next_points: Новые позиции точек (форма: N, 1, 2)
            - status: Статус отслеживания (1 = успешно, 0 = потеряно)
            - error: Ошибка отслеживания
        """
        # Преобразование в градации серого
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray1 = frame1.astype(np.float32)
            
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray2 = frame2.astype(np.float32)
        
        # Детектирование точек если не предоставлены
        if prev_points is None:
            prev_points = self.detect_features(frame1)
            if prev_points.shape[0] == 0:
                return (np.array([]).reshape(0, 1, 2), 
                       np.array([], dtype=np.uint8), 
                       np.array([], dtype=np.float32))
        
        # Преобразование точек в целые координаты
        points = prev_points.reshape(-1, 2).astype(np.int32)
        num_points = len(points)
        
        # Вычисление производных для всего изображения
        # Пространственные производные
        I_x = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
        I_y = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
        
        # Временная производная
        I_t = gray2 - gray1
        
        # Инициализация результатов
        next_points = np.zeros((num_points, 1, 2), dtype=np.float32)
        status = np.zeros(num_points, dtype=np.uint8)
        error = np.zeros(num_points, dtype=np.float32)
        
        # Размер окна (k = (window_size - 1) / 2)
        k = self.window_size // 2
        height, width = gray1.shape
        
        # Обработка каждой точки
        for i, (x, y) in enumerate(points):
            # Проверка границ
            if x < k or x >= width - k or y < k or y >= height - k:
                status[i] = 0
                continue
            
            # Извлечение окрестности
            x_min, x_max = x - k, x + k + 1
            y_min, y_max = y - k, y + k + 1
            
            I_x_window = I_x[y_min:y_max, x_min:x_max]
            I_y_window = I_y[y_min:y_max, x_min:x_max]
            I_t_window = I_t[y_min:y_max, x_min:x_max]
            
            # Вычисление сумм для системы уравнений
            # ΣI_x², ΣI_y², ΣI_x*I_y, ΣI_x*I_t, ΣI_y*I_t
            I_x2_sum = np.sum(I_x_window ** 2)
            I_y2_sum = np.sum(I_y_window ** 2)
            I_xy_sum = np.sum(I_x_window * I_y_window)
            I_xt_sum = np.sum(I_x_window * I_t_window)
            I_yt_sum = np.sum(I_y_window * I_t_window)
            
            # Матрица системы уравнений
            # [ΣI_x²   ΣI_xI_y] [u]   [ΣI_xI_t]
            # [ΣI_xI_y ΣI_y²  ] [v] = [ΣI_yI_t]
            A = np.array([[I_x2_sum, I_xy_sum],
                          [I_xy_sum, I_y2_sum]], dtype=np.float32)
            
            b = np.array([-I_xt_sum, -I_yt_sum], dtype=np.float32)
            
            # Проверка вырожденности матрицы (определитель)
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            
            # Порог для определения "хороших" точек (избегаем деления на ноль)
            # Если определитель слишком мал, точка не подходит для отслеживания
            min_det = 1e-6
            
            if abs(det) < min_det:
                # Матрица вырождена - точка не подходит для отслеживания
                status[i] = 0
                continue
            
            # Решение системы уравнений: [u, v] = A^(-1) * b
            # Обратная матрица 2x2: [a b]^(-1) = (1/det) * [d -b]
            #                        [c d]                  [-c a]
            A_inv = (1.0 / det) * np.array([[A[1, 1], -A[0, 1]],
                                            [-A[1, 0], A[0, 0]]], dtype=np.float32)
            
            flow = A_inv @ b
            
            # Проверка валидности результата (отсеивание слишком больших перемещений)
            max_flow = 50.0  # Максимальное допустимое перемещение
            if np.abs(flow[0]) > max_flow or np.abs(flow[1]) > max_flow:
                status[i] = 0
                continue
            
            # Вычисление новой позиции точки
            next_x = float(x + flow[0])
            next_y = float(y + flow[1])
            
            # Проверка границ для новой позиции
            if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
                status[i] = 0
                continue
            
            # Вычисление ошибки (brightness constancy error)
            # Ошибка = среднее значение |I_x*u + I_y*v + I_t| в окрестности
            error_window = np.abs(I_x_window * flow[0] + I_y_window * flow[1] + I_t_window)
            error[i] = np.mean(error_window)
            
            # Точка успешно отслежена
            status[i] = 1
            next_points[i, 0, 0] = next_x
            next_points[i, 0, 1] = next_y
        
        return next_points, status, error
    
    def compute_flow_vectors(self, frame1: np.ndarray, frame2: np.ndarray,
                           points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление векторов оптического потока для визуализации.
        
        Args:
            frame1: Первый кадр
            frame2: Второй кадр
            points: Начальные точки (если None, будут детектированы)
            
        Returns:
            Tuple (points, vectors, magnitudes):
            - points: Координаты точек (только успешно отслеженные)
            - vectors: Векторы потока (u, v) для каждой точки
            - magnitudes: Величины векторов
        """
        if points is None:
            points = self.detect_features(frame1)
            if points.shape[0] == 0:
                return (np.array([]).reshape(0, 2),
                       np.array([]).reshape(0, 2),
                       np.array([]))
        
        # Вычисление потока
        next_points, status, error = self.compute_flow(frame1, frame2, points)
        
        # Фильтрация успешно отслеженных точек
        good_points = status.flatten() == 1
        
        if not np.any(good_points):
            return (np.array([]).reshape(0, 2),
                   np.array([]).reshape(0, 2),
                   np.array([]))
        
        # Извлечение координат
        prev_pts = points[good_points]
        next_pts = next_points[good_points]
        
        # Вычисление векторов потока
        vectors = next_pts - prev_pts
        
        # Преобразование из формы (N, 1, 2) в (N, 2)
        prev_pts = prev_pts.reshape(-1, 2)
        vectors = vectors.reshape(-1, 2)
        
        # Вычисление величин
        magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        
        return prev_pts, vectors, magnitudes
    

