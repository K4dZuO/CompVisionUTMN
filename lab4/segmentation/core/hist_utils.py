"""
Утилиты для работы с гистограммами и сглаживанием.
"""
import numpy as np


def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Вычисляет гистограмму изображения.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        bins: int — количество бинов (по умолчанию 256)
    
    Возвращает:
        np.ndarray — гистограмма (bins элементов)
    """
    hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 256))
    return hist.astype(np.float64)


def smooth_histogram(hist: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Сглаживает гистограмму через свёртку с окном.
    
    Параметры:
        hist: np.ndarray — исходная гистограмма
        window_size: int — размер окна для свёртки (должен быть нечётным)
    
    Возвращает:
        np.ndarray — сглаженная гистограмма
    """
    if window_size % 2 == 0:
        window_size += 1
    
    # Создаём окно для свёртки (равномерное)
    window = np.ones(window_size) / window_size
    
    # Применяем свёртку с padding
    pad_size = window_size // 2
    padded_hist = np.pad(hist, pad_size, mode='edge')
    smoothed = np.convolve(padded_hist, window, mode='valid')
    
    return smoothed


def find_peaks(hist: np.ndarray, min_distance: int = 10, threshold: float = None) -> np.ndarray:
    """
    Находит пики в гистограмме.
    
    Параметры:
        hist: np.ndarray — гистограмма (возможно сглаженная)
        min_distance: int — минимальное расстояние между пиками
        threshold: float — порог для обнаружения пиков (None = автоопределение)
    
    Возвращает:
        np.ndarray — индексы пиков
    """
    if threshold is None:
        threshold = np.mean(hist) + np.std(hist)
    
    peaks = []
    n = len(hist)
    
    for i in range(min_distance, n - min_distance):
        if hist[i] > threshold:
            # Проверяем, что это локальный максимум
            is_peak = True
            for j in range(max(0, i - min_distance), min(n, i + min_distance + 1)):
                if j != i and hist[j] >= hist[i]:
                    is_peak = False
                    break
            
            if is_peak:
                # Проверяем, что не слишком близко к уже найденным пикам
                too_close = False
                for peak_idx in peaks:
                    if abs(i - peak_idx) < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    peaks.append(i)
    
    return np.array(peaks, dtype=np.int32)


def find_thresholds_between_peaks(peaks: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """
    Находит пороги между пиками как минимумы гистограммы.
    
    Параметры:
        peaks: np.ndarray — индексы пиков
        hist: np.ndarray — гистограмма
    
    Возвращает:
        np.ndarray — пороги (индексы минимумов между пиками)
    """
    if len(peaks) < 2:
        return np.array([], dtype=np.int32)
    
    peaks_sorted = np.sort(peaks)
    thresholds = []
    
    for i in range(len(peaks_sorted) - 1):
        start = peaks_sorted[i]
        end = peaks_sorted[i + 1]
        
        # Находим минимум между пиками
        min_idx = start + np.argmin(hist[start:end])
        thresholds.append(min_idx)
    
    return np.array(thresholds, dtype=np.int32)

