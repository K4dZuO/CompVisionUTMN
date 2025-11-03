"""
Оптимизированные версии функций сегментации с использованием векторных операций,
FFT, сепарабельных свёрток и других математических оптимизаций.
"""
import numpy as np
from typing import Optional


def apply_sobel_optimized(image: np.ndarray, use_fp16: bool = False):
    """
    Оптимизированная версия оператора Собеля с использованием сепарабельных свёрток.
    
    Ядро Собеля сепарабельно: K = Kx_1D * Ky_1D^T, где
    Kx_1D = [1, 2, 1]
    Ky_1D = [-1, 0, 1]
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого
        use_fp16: bool — использовать float16 вместо float64
    
    Возвращает:
        tuple: (Gx, Gy) — градиенты по X и Y
    """
    if use_fp16:
        dtype = np.float16
    else:
        dtype = np.float32  # Используем float32 вместо float64 для ускорения
    
    image_float = image.astype(dtype)
    h, w = image_float.shape
    
    # Сепарабельные 1D ядра
    kernel_x_1d = np.array([1, 2, 1], dtype=dtype) / 4.0  # Нормализация
    kernel_y_1d = np.array([-1, 0, 1], dtype=dtype)
    
    # Padding
    padded = np.pad(image_float, ((1, 1), (1, 1)), mode='edge')
    
    # Gx: сначала вертикальная свёртка, затем горизонтальная
    # Gx = I * (Kx_1D * Ky_1D^T)
    # = I * (вертикальная свёртка) * (горизонтальная свёртка)
    
    # Для Gx: сначала применяем вертикальную свёртку с kernel_y_1d
    temp = np.zeros((h, w + 2), dtype=dtype)
    for i in range(h):
        for j in range(1, w + 1):
            temp[i, j] = (padded[i, j-1] * kernel_y_1d[0] + 
                          padded[i+1, j-1] * kernel_y_1d[1] + 
                          padded[i+2, j-1] * kernel_y_1d[2])
    
    # Затем горизонтальная свёртка с kernel_x_1d
    Gx = np.zeros((h, w), dtype=dtype)
    for i in range(h):
        for j in range(w):
            Gx[i, j] = (temp[i, j] * kernel_x_1d[0] + 
                        temp[i, j+1] * kernel_x_1d[1] + 
                        temp[i, j+2] * kernel_x_1d[2])
    
    # Для Gy: применяем в обратном порядке
    # Gy: сначала горизонтальная, затем вертикальная
    temp = np.zeros((h + 2, w), dtype=dtype)
    for i in range(1, h + 1):
        for j in range(w):
            temp[i, j] = (padded[i-1, j] * kernel_x_1d[0] + 
                          padded[i-1, j+1] * kernel_x_1d[1] + 
                          padded[i-1, j+2] * kernel_x_1d[2])
    
    Gy = np.zeros((h, w), dtype=dtype)
    for i in range(h):
        for j in range(w):
            Gy[i, j] = (temp[i, j] * kernel_y_1d[0] + 
                        temp[i+1, j] * kernel_y_1d[1] + 
                        temp[i+2, j] * kernel_y_1d[2])
    
    return Gx, Gy


def apply_sobel_vectorized(image: np.ndarray, use_fp16: bool = False):
    """
    Полностью векторизованная версия оператора Собеля через strided views.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого
        use_fp16: bool — использовать float16 вместо float64
    
    Возвращает:
        tuple: (Gx, Gy) — градиенты по X и Y
    """
    if use_fp16:
        dtype = np.float16
    else:
        dtype = np.float32
    
    image_float = image.astype(dtype)
    h, w = image_float.shape
    pad_size = 1
    
    # Padding
    padded = np.pad(image_float, pad_size, mode='edge')
    
    # Ядра Собеля
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=dtype)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=dtype)
    
    # Попытка использовать sliding_window_view (numpy >= 1.20)
    try:
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        # Векторизованная свёртка
        Gx = np.sum(windows * sobel_x, axis=(2, 3))
        Gy = np.sum(windows * sobel_y, axis=(2, 3))
    except (AttributeError, ValueError):
        # Fallback на ручную векторизацию
        Gx = np.zeros_like(image_float, dtype=dtype)
        Gy = np.zeros_like(image_float, dtype=dtype)
        
        for i in range(h):
            for j in range(w):
                region = padded[i:i+3, j:j+3]
                Gx[i, j] = np.sum(region * sobel_x)
                Gy[i, j] = np.sum(region * sobel_y)
    
    return Gx, Gy


def apply_roberts_vectorized(image: np.ndarray, use_fp16: bool = False):
    """
    Векторизованная версия оператора Робертса.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого
        use_fp16: bool — использовать float16 вместо float64
    
    Возвращает:
        tuple: (Gx, Gy) — градиенты по X и Y
    """
    if use_fp16:
        dtype = np.float16
    else:
        dtype = np.float32
    
    image_float = image.astype(dtype)
    h, w = image_float.shape
    
    # Векторизованное вычисление через срезы
    # Gx[i,j] = I[i,j] - I[i+1,j+1]
    # Gy[i,j] = I[i,j+1] - I[i+1,j]
    
    Gx = image_float[:-1, :-1] - image_float[1:, 1:]
    Gy = image_float[:-1, 1:] - image_float[1:, :-1]
    
    # Добавляем нулевую границу для совместимости
    Gx = np.pad(Gx, ((0, 1), (0, 1)), mode='constant')
    Gy = np.pad(Gy, ((0, 1), (0, 1)), mode='constant')
    
    return Gx, Gy


def threshold_iterative_vectorized(image: np.ndarray, eps: float = 0.5, max_iter: int = 50, use_fp16: bool = False):
    """
    Векторизованная версия итеративного метода пороговой сегментации.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        eps: float — критерий останова
        max_iter: int — максимум итераций
        use_fp16: bool — использовать float16 для вычислений
    
    Возвращает:
        tuple: (T, binary_mask) — порог и бинарная маска (0 и 255)
    """
    if use_fp16:
        dtype = np.float16
    else:
        dtype = np.float32
    
    image_float = image.astype(dtype)
    flat = image_float.flatten()
    
    # Начальное приближение
    T_old = dtype((np.min(flat) + np.max(flat)) / 2.0)
    T_new = T_old
    
    for iteration in range(max_iter):
        # Векторизованное разделение на классы
        mask1 = flat <= T_old
        mask2 = ~mask1
        
        class1 = flat[mask1]
        class2 = flat[mask2]
        
        if len(class1) == 0 or len(class2) == 0:
            T_new = T_old
            break
        
        # Векторизованное вычисление средних
        mu1 = np.mean(class1)
        mu2 = np.mean(class2)
        
        # Новый порог
        T_new = dtype((mu1 + mu2) / 2.0)
        
        # Проверка останова
        if abs(T_new - T_old) < eps:
            break
        
        T_old = T_new
    
    T = T_new
    
    # Векторизованная бинаризация
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > T] = 255
    
    return float(T), binary


def kmeans_1d_vectorized(data: np.ndarray, k: int = 2, max_iter: int = 100, eps: float = 0.1, use_fp16: bool = False):
    """
    Векторизованная версия K-средних для 1D-кластеризации.
    
    Параметры:
        data: np.ndarray — одномерный массив значений яркости
        k: int — количество кластеров
        max_iter: int — максимум итераций
        eps: float — критерий останова
        use_fp16: bool — использовать float16 для вычислений
    
    Возвращает:
        tuple: (centroids, labels) — центроиды и метки кластеров
    """
    if use_fp16:
        dtype = np.float16
    else:
        dtype = np.float32
    
    data_float = data.astype(dtype)
    n = len(data_float)
    
    # Инициализация центроидов
    min_val, max_val = np.min(data_float), np.max(data_float)
    centroids = np.linspace(min_val, max_val, k, dtype=dtype)
    
    labels = np.zeros(n, dtype=np.int32)
    
    for iteration in range(max_iter):
        # Векторизованное присваивание меток через broadcasting
        # distances[i, j] = |data[i] - centroids[j]|
        distances = np.abs(data_float[:, np.newaxis] - centroids[np.newaxis, :])
        labels = np.argmin(distances, axis=1)
        
        # Векторизованное обновление центроидов
        centroids_new = np.zeros(k, dtype=dtype)
        for j in range(k):
            cluster_points = data_float[labels == j]
            if len(cluster_points) > 0:
                centroids_new[j] = np.mean(cluster_points)
            else:
                centroids_new[j] = centroids[j]
        
        # Проверка останова
        if np.all(np.abs(centroids_new - centroids) < eps):
            break
        
        centroids = centroids_new
    
    return centroids, labels


def adaptive_threshold_vectorized(image: np.ndarray, window_size: int = 15, 
                                  stat_type: str = 'mean', C: float = 0.0, use_fp16: bool = False):
    """
    Векторизованная версия адаптивной пороговой сегментации.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        window_size: int — размер окна (должен быть нечётным)
        stat_type: str — тип статистики ('mean', 'median', 'avg_min_max')
        C: float — корректирующий параметр
        use_fp16: bool — использовать float16 для вычислений
    
    Возвращает:
        np.ndarray — бинарная маска (0 и 255)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    if use_fp16:
        dtype = np.float16
    else:
        dtype = np.float32
    
    h, w = image.shape
    half_window = window_size // 2
    
    # Padding
    padded = np.pad(image.astype(dtype), half_window, mode='edge')
    
    # Векторизованное вычисление локальной статистики
    if stat_type == 'mean':
        # Используем uniform_filter для быстрого вычисления среднего
        from scipy.ndimage import uniform_filter
        try:
            local_stats = uniform_filter(padded.astype(np.float64), size=window_size, mode='constant')
            local_stats = local_stats[half_window:-half_window, half_window:-half_window]
            local_stats = local_stats.astype(dtype)
        except ImportError:
            # Fallback на ручную реализацию
            local_stats = np.zeros_like(image, dtype=dtype)
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+window_size, j:j+window_size]
                    local_stats[i, j] = np.mean(window)
    elif stat_type == 'median':
        from scipy.ndimage import median_filter
        try:
            local_stats = median_filter(padded.astype(np.float64), size=window_size, mode='constant')
            local_stats = local_stats[half_window:-half_window, half_window:-half_window]
            local_stats = local_stats.astype(dtype)
        except ImportError:
            # Fallback на ручную реализацию
            local_stats = np.zeros_like(image, dtype=dtype)
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+window_size, j:j+window_size]
                    local_stats[i, j] = np.median(window)
    elif stat_type == 'avg_min_max':
        # Используем максимальный и минимальный фильтры
        from scipy.ndimage import maximum_filter, minimum_filter
        try:
            max_vals = maximum_filter(padded.astype(np.float64), size=window_size, mode='constant')
            min_vals = minimum_filter(padded.astype(np.float64), size=window_size, mode='constant')
            max_vals = max_vals[half_window:-half_window, half_window:-half_window]
            min_vals = min_vals[half_window:-half_window, half_window:-half_window]
            local_stats = ((max_vals + min_vals) / 2.0).astype(dtype)
        except ImportError:
            # Fallback на ручную реализацию
            local_stats = np.zeros_like(image, dtype=dtype)
            for i in range(h):
                for j in range(w):
                    window = padded[i:i+window_size, j:j+window_size]
                    local_stats[i, j] = (np.min(window) + np.max(window)) / 2.0
    else:
        raise ValueError(f"Неизвестный тип статистики: {stat_type}")
    
    # Векторизованная бинаризация
    thresholds = local_stats - dtype(C)
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > thresholds] = 255
    
    return binary


def smooth_histogram_vectorized(hist: np.ndarray, window_size: int = 5):
    """
    Векторизованная версия сглаживания гистограммы.
    
    Параметры:
        hist: np.ndarray — исходная гистограмма
        window_size: int — размер окна для свёртки
    
    Возвращает:
        np.ndarray — сглаженная гистограмма
    """
    if window_size % 2 == 0:
        window_size += 1
    
    # Создаём окно для свёртки
    window = np.ones(window_size) / window_size
    
    # Применяем свёртку через numpy.convolve
    pad_size = window_size // 2
    padded_hist = np.pad(hist, pad_size, mode='edge')
    smoothed = np.convolve(padded_hist, window, mode='valid')
    
    return smoothed


def edge_segmentation_optimized(image: np.ndarray, operator: str = 'sobel', 
                                 T_edge: float = 50.0, use_fp16: bool = False):
    """
    Оптимизированная версия сегментации по краям.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        operator: str — оператор ('sobel', 'prewitt', 'roberts')
        T_edge: float — порог для бинаризации градиента
        use_fp16: bool — использовать float16 для вычислений
    
    Возвращает:
        np.ndarray — бинарная маска краёв (0 и 255)
    """
    # Выбор оператора
    if operator.lower() == 'sobel':
        Gx, Gy = apply_sobel_vectorized(image, use_fp16=use_fp16)
    elif operator.lower() == 'prewitt':
        # Для Превитта используем аналогичную векторизацию
        from segmentation.core.edges import apply_prewitt
        Gx, Gy = apply_prewitt(image)
    elif operator.lower() == 'roberts':
        Gx, Gy = apply_roberts_vectorized(image, use_fp16=use_fp16)
    else:
        raise ValueError(f"Неизвестный оператор: {operator}")
    
    # Вычисление модуля градиента (векторизовано)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    # Векторизованная бинаризация
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[magnitude > T_edge] = 255
    
    return binary

