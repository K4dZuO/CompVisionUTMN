import numpy as np
from scipy import ndimage
from typing import Tuple, Optional

def logarithmic_transform(image: np.ndarray) -> np.ndarray:
    """Логарифмическое преобразование изображения."""
    image_float = image.astype(np.float64) + 1
    c = 255 / np.log(1 + np.max(image_float))
    transformed = c * np.log(image_float)
    return np.clip(transformed, 0, 255).astype(np.uint8)

def power_transform(image: np.ndarray, gamma: float) -> np.ndarray:
    """Степенное преобразование изображения с произвольным значением гаммы."""
    image_float = image.astype(np.float64) / 255.0
    c = 1.0
    transformed = c * (image_float ** gamma)
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
    """Прямоугольный фильтр (усредняющий фильтр)."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return ndimage.convolve(image.astype(np.float64), kernel).astype(np.uint8)

def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Медианный фильтр."""
    return ndimage.median_filter(image, size=kernel_size)

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Создание гауссова ядра."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    return kernel / np.sum(kernel)

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Фильтр Гаусса с размером ядра, определенным правилом 3σ."""
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(kernel_size, sigma)
    return ndimage.convolve(image.astype(np.float64), kernel).astype(np.uint8)

def sigma_filter(image: np.ndarray, sigma: float, window_size: int = 5) -> np.ndarray:
    """Сигма-фильтр."""
    pad = window_size // 2
    image_padded = np.pad(image, pad, mode='reflect')
    result = np.zeros_like(image, dtype=np.float64)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = image_padded[i:i+window_size, j:j+window_size]
            center_val = image[i, j]
            mask = np.abs(window - center_val) <= sigma
            if np.any(mask):
                result[i, j] = np.mean(window[mask])
            else:
                result[i, j] = center_val
    
    return np.clip(result, 0, 255).astype(np.uint8)

def absolute_difference_map(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Карта абсолютной разности между двумя изображениями."""
    diff = np.abs(image1.astype(np.float64) - image2.astype(np.float64))
    return np.clip(diff, 0, 255).astype(np.uint8)

def unsharp_masking(image: np.ndarray, k: int, lambda_val: float) -> np.ndarray:
    """Нерезкое маскирование для повышения резкости изображения."""
    blurred = gaussian_filter(image, sigma=k/3)
    mask = image.astype(np.float64) - blurred.astype(np.float64)
    sharpened = image.astype(np.float64) + lambda_val * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """Добавление гауссова шума к изображению."""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image.astype(np.float64) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image: np.ndarray, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> np.ndarray:
    """Добавление солевого-перечного шума к изображению."""
    noisy_image = image.copy()
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_image[salt_mask] = 255
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0
    return noisy_image