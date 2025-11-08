"""
Адаптивная пороговая сегментация.
"""
import numpy as np


def compute_local_statistic(window: np.ndarray, stat_type: str = 'mean') -> float:
    """
    Вычисляет локальную статистику для окна.
    
    Параметры:
        window: np.ndarray — окно изображения
        stat_type: str — тип статистики ('mean', 'median', 'avg_min_max')
    
    Возвращает:
        float — значение статистики
    """
    if stat_type == 'mean':
        return np.mean(window)
    elif stat_type == 'median':
        return np.median(window)
    elif stat_type == 'avg_min_max':
        return (np.min(window) + np.max(window)) / 2.0
    else:
        raise ValueError(f"Неизвестный тип статистики: {stat_type}")


def adaptive_threshold(image: np.ndarray, window_size: int = 15, 
                       stat_type: str = 'mean', C: float = 0.0) -> np.ndarray:
    """
    Адаптивная пороговая сегментация по локальной статистике.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        window_size: int — размер окна (должен быть нечётным)
        stat_type: str — тип статистики ('mean', 'median', 'avg_min_max')
        C: float — корректирующий параметр (вычитается из порога)
    
    Возвращает:
        np.ndarray — бинарная маска (0 и 255)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    h, w = image.shape
    binary = np.zeros_like(image, dtype=np.uint8)
    half_window = window_size // 2
    
    # Padding для обработки границ
    padded = np.pad(image, half_window, mode='edge')
    
    for i in range(h):
        for j in range(w):
            # Извлекаем окно
            window = padded[i:i + window_size, j:j + window_size]
            
            # Вычисляем локальную статистику
            local_stat = compute_local_statistic(window, stat_type)
            
            # Вычисляем порог
            T_local = local_stat - C
            
            # Бинаризация
            if image[i, j] > T_local:
                binary[i, j] = 255
    
    return binary


def adaptive_threshold_optimized(image: np.ndarray, window_size: int = 15,
                                 stat_type: str = 'mean', C: float = 0.0) -> np.ndarray:
    """
    Оптимизированная версия адаптивного порога (быстрее для больших изображений).
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        window_size: int — размер окна (должен быть нечётным)
        stat_type: str — тип статистики ('mean', 'median', 'avg_min_max')
        C: float — корректирующий параметр (вычитается из порога)
    
    Возвращает:
        np.ndarray — бинарная маска (0 и 255)
    """
    if window_size % 2 == 0:
        window_size += 1
    
    h, w = image.shape
    half_window = window_size // 2
    
    # Padding
    padded = np.pad(image.astype(np.float64), half_window, mode='edge')
    
    # Вычисляем карту локальных статистик
    if stat_type == 'mean':
        # Используем свёртку для среднего
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
        local_stats = np.zeros_like(image, dtype=np.float64)
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i + window_size, j:j + window_size]
                local_stats[i, j] = np.mean(window)
    elif stat_type == 'median':
        local_stats = np.zeros_like(image, dtype=np.float64)
        for i in range(h):
            for j in range(w):
                window = padded[i:i + window_size, j:j + window_size]
                local_stats[i, j] = np.median(window)
    elif stat_type == 'avg_min_max':
        local_stats = np.zeros_like(image, dtype=np.float64)
        for i in range(h):
            for j in range(w):
                window = padded[i:i + window_size, j:j + window_size]
                local_stats[i, j] = (np.min(window) + np.max(window)) / 2.0
    else:
        raise ValueError(f"Неизвестный тип статистики: {stat_type}")
    
    # Бинаризация
    thresholds = local_stats - C
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > thresholds] = 255
    
    return binary

