import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
from numpy.lib.stride_tricks import sliding_window_view


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Оптимизированная свертка 2D с использованием векторизованных операций."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    windows = sliding_window_view(padded, (kh, kw))
    # Используем einsum для более эффективного вычисления свертки
    result = np.einsum('ijkl,kl->ij', windows, kernel)
    return result
    
def apply_per_channel(image: np.ndarray, func, *args, **kwargs):
    """Применяет фильтр func к каждому каналу RGB отдельно."""
    if image.ndim == 2:
        return func(image, *args, **kwargs)
    elif image.ndim == 3:
        # Берем только первые 3 канала, если вдруг ARGB/BGRA
        image = image[:, :, :3]
        channels = [func(image[:, :, i], *args, **kwargs) for i in range(image.shape[2])]
        return np.stack(channels, axis=2)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")




def logarithmic_transform(image: np.ndarray) -> np.ndarray:
    """Логарифмическое преобразование изображения - оптимизированная версия."""
    # Используем float32 вместо float64 для экономии памяти
    image_float = image.astype(np.float32) + 1
    c = 255.0 / np.log(1.0 + np.max(image_float))
    transformed = c * np.log(image_float)
    return np.clip(transformed, 0, 255).astype(np.uint8)

def power_transform(image: np.ndarray, gamma: float) -> np.ndarray:
    """Степенное преобразование изображения с произвольным значением гаммы - оптимизированная версия."""
    # Используем float32 вместо float64 для экономии памяти
    image_float = image.astype(np.float32) / 255.0
    transformed = image_float ** gamma
    return np.clip(transformed * 255, 0, 255).astype(np.uint8)

def binary_transform(image: np.ndarray, threshold: int) -> np.ndarray:
    """Бинарное преобразование с произвольным пороговым значением."""
    binary = np.zeros_like(image)
    binary[image >= threshold] = 255
    return binary

def brightness_range_cutout(image: np.ndarray, min_val: int, max_val: int, constant_value: int = None) -> np.ndarray:
    """Вырезание произвольного диапазона яркостей."""
    mask = (image >= min_val) & (image <= max_val)
    
    if constant_value is not None:
        result = np.full_like(image, constant_value)
        result[mask] = image[mask]
    else:
        result = image.copy()
        result[~mask] = 0
    
    return result

def rectangular_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Прямоугольный усредняющий фильтр без SciPy - оптимизированная версия."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    def _apply(gray):
        return convolve2d(gray.astype(np.float32), kernel)

    return np.clip(apply_per_channel(image, _apply), 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Медианный фильтр с numpy без циклов."""
    from numpy.lib.stride_tricks import sliding_window_view

    def _median(gray):
        pad = kernel_size // 2
        padded = np.pad(gray, pad, mode='reflect')
        # формируем "окна" вокруг каждого пикселя
        windows = sliding_window_view(padded, (kernel_size, kernel_size)) #трюк перевода в векторные операции для ускорения
        # вычисляем медиану по последним двум осям (окно)
        return np.median(windows, axis=(-2, -1))
    
    return apply_per_channel(image, _median).astype(np.uint8)

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Создание гауссова ядра - оптимизированная версия."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    # Используем более эффективное вычисление
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссов фильтр без SciPy - оптимизированная версия."""
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(kernel_size, sigma).astype(np.float32)
    def _apply(gray):
        return convolve2d(gray.astype(np.float32), kernel)
    return np.clip(apply_per_channel(image, _apply), 0, 255).astype(np.uint8)


def sigma_filter(image: np.ndarray, sigma: float, window_size: int = 5) -> np.ndarray:
    """Сигма-фильтр для RGB и grayscale - оптимизированная версия."""
    pad = window_size // 2

    def _apply(gray):
        padded = np.pad(gray, pad, mode='reflect')
        # Создаем окна для всех пикселей одновременно
        windows = sliding_window_view(padded, (window_size, window_size))
        
        # Вычисляем стандартное отклонение для каждого окна
        std_local = np.std(windows, axis=(-2, -1))
        
        # Создаем маски для всех пикселей одновременно
        center_vals = gray.astype(np.float32)
        center_vals_expanded = center_vals[:, :, np.newaxis, np.newaxis]
        diff = np.abs(windows - center_vals_expanded)
        threshold = np.maximum(sigma, std_local)
        threshold_expanded = threshold[:, :, np.newaxis, np.newaxis]
        mask = diff <= threshold_expanded
        
        # Вычисляем среднее для каждого окна с учетом маски
        result = np.zeros_like(gray, dtype=np.float32)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if np.any(mask[i, j]):
                    result[i, j] = np.mean(windows[i, j][mask[i, j]])
                else:
                    result[i, j] = center_vals[i, j]
        
        return result

    return apply_per_channel(image, _apply).astype(np.uint8)


def absolute_difference_map(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Карта абсолютной разности между двумя изображениями - оптимизированная версия."""
    # Используем float32 вместо float64 для экономии памяти
    diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
    return np.clip(diff, 0, 255).astype(np.uint8)

def unsharp_masking(image: np.ndarray, k: int, lambda_val: float) -> np.ndarray:
    """Нерезкое маскирование для повышения резкости изображения - оптимизированная версия."""
    blurred = gaussian_filter(image, sigma=k/3)
    # Используем float32 вместо float64 для экономии памяти
    image_float = image.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    mask = image_float - blurred_float
    sharpened = image_float + lambda_val * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """Добавление гауссова шума к изображению - оптимизированная версия."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image: np.ndarray, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> np.ndarray:
    """Добавление солевого-перечного шума к изображению."""
    noisy_image = image.copy()
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_image[salt_mask] = 255
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0
    return noisy_image