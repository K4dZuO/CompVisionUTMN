"""
Реализация алгоритма Хорна-Шанка для вычисления плотного оптического потока.

АЛГОРИТМ ХОРНА-ШАНКА:
====================

Математическая основа:
----------------------
Алгоритм решает задачу оптического потока на основе двух предположений:

1. Brightness Constancy Constraint (BCC):
   Яркость точки не меняется при движении:
   I(x, y, t) = I(x + u, y + v, t + 1)
   
   Линеаризация первого порядка приводит к уравнению:
   I_x*u + I_y*v + I_t = 0
   
   где:
   - I_x, I_y - пространственные производные яркости
   - I_t - временная производная
   - u, v - компоненты вектора оптического потока (горизонтальная, вертикальная)

2. Smoothness Constraint:
   Оптический поток должен быть гладким (соседние точки движутся похоже).
   Это позволяет разрешить aperture problem (неоднозначность направления движения).

Энергетический функционал:
--------------------------
Алгоритм минимизирует функционал:
E = ∫∫[(I_x*u + I_y*v + I_t)² + λ²(‖∇u‖² + ‖∇v‖²)]dxdy

где:
- Первый член: ошибка brightness constancy constraint
- Второй член: регуляризация гладкости (smoothness term)
- λ: весовой коэффициент, балансирующий между точностью и гладкостью

Решение:
--------
Минимизация функционала приводит к системе дифференциальных уравнений Эйлера-Лагранжа,
которая решается итерационным методом (метод Якоби или Гаусса-Зейделя):

u^(n+1) = u̅^(n) - (I_x*(I_x*u̅^(n) + I_y*v̅^(n) + I_t)) / (λ² + I_x² + I_y²)
v^(n+1) = v̅^(n) - (I_y*(I_x*u̅^(n) + I_y*v̅^(n) + I_t)) / (λ² + I_x² + I_y²)

где u̅, v̅ - локальные средние значений u и v.

Параметры:
----------
- lambda_val: весовой коэффициент регуляризации (больше значение = более гладкий поток)
- num_iterations: количество итераций для сходимости
- threshold: порог для определения сходимости (опционально)

Преимущества:
-------------
- Плотный поток (вектор для каждого пикселя)
- Устойчивость к шуму благодаря регуляризации
- Хорошо работает для медленных движений

Недостатки:
-----------
- Вычислительно затратный (итеративный процесс)
- Может сглаживать резкие границы движения
- Требует тщательного подбора параметра λ
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class HornSchunckProcessor:
    """
    Процессор для вычисления оптического потока методом Хорна-Шанка.
    
    Оптимизации:
    - Использование векторизованных операций NumPy
    - Предвычисление производных и их квадратов
    - Кэширование знаменателя для ускорения итераций
    - Поддержка многоядерных вычислений через NumPy
    """
    
    def __init__(self, lambda_val: float = 1.0, num_iterations: int = 100, 
                 threshold: float = 0.001):
        """
        Инициализация процессора Хорна-Шанка.
        
        Args:
            lambda_val: Весовой коэффициент регуляризации (больше = более гладкий поток)
            num_iterations: Количество итераций для сходимости
            threshold: Порог сходимости T - если среднее изменение между итерациями
                      меньше этого значения, итерации останавливаются досрочно.
                      Если threshold <= 0, используется фиксированное количество итераций.
        """
        self.lambda_val = lambda_val
        self.num_iterations = num_iterations
        self.threshold = threshold
        
        # Кэш для производных (оптимизация)
        self._I_x_cache = None
        self._I_y_cache = None
        self._I_t_cache = None
        self._denominator_cache = None
        self._prev_frame = None
        
    def compute_derivatives(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление пространственных и временной производных.
        
        Использует оператор Собеля для пространственных производных
        и простое вычитание для временной производной.
        
        Args:
            frame1: Первый кадр (градации серого)
            frame2: Второй кадр (градации серого)
            
        Returns:
            Tuple (I_x, I_y, I_t): Производные по x, y и времени
        """
        # Преобразование в float32 для точности вычислений
        frame1 = frame1.astype(np.float32)
        frame2 = frame2.astype(np.float32)
        
        # Пространственные производные используя оператор Собеля
        # ksize=3 дает хороший баланс между точностью и устойчивостью к шуму
        I_x = cv2.Sobel(frame1, cv2.CV_32F, 1, 0, ksize=3)
        I_y = cv2.Sobel(frame1, cv2.CV_32F, 0, 1, ksize=3)
        
        # Временная производная (разность между кадрами)
        I_t = frame2 - frame1
        
        return I_x, I_y, I_t
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление оптического потока методом Хорна-Шанка.
        
        Алгоритм:
        1. Вычисление производных I_x, I_y, I_t
        2. Инициализация u, v нулями
        3. Итеративное обновление u, v используя локальные средние
        4. Сходимость к решению методом Якоби
        
        Args:
            frame1: Первый кадр (градации серого или RGB)
            frame2: Второй кадр (градации серого или RGB)
            
        Returns:
            Tuple (u, v): Компоненты оптического потока (горизонтальная, вертикальная)
        """
        # Преобразование в градации серого если необходимо
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Вычисление производных
        I_x, I_y, I_t = self.compute_derivatives(frame1, frame2)
        
        # Инициализация потоков нулями
        height, width = frame1.shape
        u = np.zeros((height, width), dtype=np.float32)
        v = np.zeros((height, width), dtype=np.float32)
        
        # Предвычисление знаменателя для оптимизации
        # Знаменатель не меняется между итерациями
        denominator = self.lambda_val**2 + I_x**2 + I_y**2
        # Избегаем деления на ноль
        denominator = np.maximum(denominator, 1e-10)
        
        # Итеративное решение методом Якоби
        # Используем ядро усреднения 3x3 для вычисления локальных средних
        kernel = np.array([[0, 0.25, 0],
                          [0.25, 0, 0.25],
                          [0, 0.25, 0]], dtype=np.float32)
        
        # Предыдущие значения для проверки сходимости
        u_prev = np.zeros_like(u)
        v_prev = np.zeros_like(v)
        
        for iteration in range(self.num_iterations):
            # Вычисление локальных средних (лапласиан)
            # Используем фильтрацию для эффективного вычисления
            u_avg = cv2.filter2D(u, -1, kernel)
            v_avg = cv2.filter2D(v, -1, kernel)
            
            # Обновление u и v согласно уравнениям Хорна-Шанка
            # P = I_x*u_avg + I_y*v_avg + I_t
            P = I_x * u_avg + I_y * v_avg + I_t
            
            # Обновление компонент потока
            u = u_avg - (I_x * P) / denominator
            v = v_avg - (I_y * P) / denominator
            
            # Проверка сходимости по порогу T
            # Вычисляем среднее изменение между итерациями
            if self.threshold > 0:
                u_change = np.mean(np.abs(u - u_prev))
                v_change = np.mean(np.abs(v - v_prev))
                max_change = max(u_change, v_change)
                
                # Если изменение меньше порога, останавливаем итерации
                if max_change < self.threshold:
                    break
            
            # Сохранение текущих значений для следующей итерации
            u_prev = u.copy()
            v_prev = v.copy()
        
        return u, v
    
    def compute_flow_magnitude_direction(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление оптического потока с дополнительными метриками.
        
        Returns:
            Tuple (u, v, magnitude, angle):
            - u, v: компоненты потока
            - magnitude: величина потока (скорость движения)
            - angle: направление потока в радианах
        """
        u, v = self.compute_flow(frame1, frame2)
        
        # Вычисление величины и направления
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(v, u)
        
        return u, v, magnitude, angle
    
    def set_parameters(self, lambda_val: Optional[float] = None, 
                      num_iterations: Optional[int] = None,
                      threshold: Optional[float] = None):
        """
        Обновление параметров алгоритма.
        
        Args:
            lambda_val: Новое значение lambda (если не None)
            num_iterations: Новое количество итераций (если не None)
            threshold: Новый порог (если не None)
        """
        if lambda_val is not None:
            self.lambda_val = lambda_val
        if num_iterations is not None:
            self.num_iterations = num_iterations
        if threshold is not None:
            self.threshold = threshold
        
        # Очистка кэша при изменении параметров
        self._I_x_cache = None
        self._I_y_cache = None
        self._I_t_cache = None
        self._denominator_cache = None

