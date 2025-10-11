import numpy as np
from math import ceil
from numpy.lib.stride_tricks import sliding_window_view


# def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
#     """Оптимизированная свертка 2D с использованием векторизованных операций."""
#     kh, kw = kernel.shape
#     pad_h, pad_w = kh // 2, kw // 2
#     padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
#     windows = sliding_window_view(padded, (kh, kw))
#     # Используем einsum для более эффективного вычисления свертки
#     result = np.einsum('ijkl,kl->ij', windows, kernel)
#     return result
    
# def apply_per_channel(image: np.ndarray, func, *args, **kwargs):
#     """Применяет фильтр func к каждому каналу RGB отдельно."""
#     if image.ndim == 2:
#         return func(image, *args, **kwargs)
#     elif image.ndim == 3:
#         # Берем только первые 3 канала, если вдруг ARGB/BGRA
#         image = image[:, :, :3]
#         channels = [func(image[:, :, i], *args, **kwargs) for i in range(image.shape[2])]
#         return np.stack(channels, axis=2)
#     else:
#         raise ValueError(f"Unexpected image shape: {image.shape}")

def logarithmic_transform(image: np.ndarray) -> np.ndarray:
    """Логарифмическое преобразование изображения - оптимизированная версия."""
    image_float = image.astype(np.float32) + 1
    c = 255.0 / np.log(1 + np.max(image_float))
    transformed = c * np.log(1 + image_float)
    return np.clip(transformed, 0, 255).astype(np.uint8) # все что x < 0 = 0, x > 255 = 255

def power_transform(image: np.ndarray, gamma: float) -> np.ndarray:
    """Степенное преобразование изображения с произвольным значением гаммы - оптимизированная версия."""
    image_float = image.astype(np.float32) / 255
    c = 255 / np.log(1 + np.max(image_float))
    transformed = c* image_float**gamma
    return np.clip(transformed, 0, 255).astype(np.uint8)

def binary_transform(image: np.ndarray, threshold: int) -> np.ndarray:
    """Бинарное преобразование с произвольным пороговым значением."""
    binary = np.zeros_like(image)
    binary[image >= threshold] = 255
    return binary

def brightness_range_cutout(image: np.ndarray, min_val: int, max_val: int, constant_value: int = None) -> np.ndarray:
    """Вырезание произвольного диапазона яркостей."""
    mask = (image >= min_val) & (image <= max_val) # маска пикселей внутри диапазона
    if constant_value is not None:
        result = np.full_like(image, constant_value) # Массив той же формы с одним значением
        result[mask] = image[mask] 
    else:
        result = image
            
    return result

def rectangular_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    n, m, channels = image.shape
    print(image.shape)
    print(image.ndim)
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge') # добавляем по краям, но не в каналы
    filter_image = np.zeros_like(image)
    for c in range(channels):
        for i in range(n):
            for j in range(m):
                kernel = padded_image[i:i+kernel_size,j:j+kernel_size, c]
                filter_image[i, j, c] = np.average(kernel)
        print(f"Обработка: {ceil(100 * (c+1) / channels)}%")
    print(filter_image)
    return np.clip(filter_image, 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    n, m, channels = image.shape
    
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge') # добавляем по краям, но не в каналы
    filter_image = np.zeros_like(image)
    for c in range(channels):
        for i in range(n):
            for j in range(m):
                kernel = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                filter_image[i, j, c] = np.median(kernel)
        
        print(f"Обработка: {ceil(100 * (c+1) / channels)}%")
    return np.clip(filter_image, 0, 255).astype(np.uint8)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Создаёт гауссово ядро размером size x size.
    """
    kernel = np.zeros((size, size), dtype=np.float64) # условно матрица весов
    center = size // 2 
    s = 2 * (sigma ** 2)
    total = 0.0

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center # расстояние от центра
            val = np.exp(-(x**2 + y**2) / s)
            kernel[i, j] = val
            total += val

    kernel /= total  # Нормализация
    return kernel

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # Правило 3*sigma
    kernel = gaussian_kernel(kernel_size, sigma)

    n, m, channels = image.shape
    
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge') # добавляем по краям, но не в каналы
    filter_image = np.zeros_like(image, dtype=np.float64)

    for c in range(channels):
        for i in range(n):
            for j in range(m):
                region = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                filter_image[i, j, c] = np.sum(region * kernel) # 
        print(f"Обработка: {ceil(100 * (c+1) / channels)}%")
    return filter_image.astype(np.uint8)

def sigma_filter(image: np.ndarray, sigma: float, window_size: int) -> np.ndarray:
    n, m, channels = image.shape
    pad = window_size//2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge') # добавляем по краям, но не в каналы
    filter_image = np.zeros_like(image, dtype=np.float64)

    for c in range(channels):
        for i in range(n):
            for j in range(m):
                center_val = padded_image[i + pad, j + pad, c] #  яркость центрального пикселя.
                total_val = 0.0 
                total_weight = 0.0 
                for di in range(-pad, pad + 1): # ходим в окне
                    for dj in range(-pad, pad + 1):
                        neighbor_val = padded_image[i + pad + di, j + pad + dj, c]
                        if abs(center_val - neighbor_val) <= sigma: # если разница меньше сигмы
                            total_val += neighbor_val
                            total_weight += 1
                if total_weight > 0:
                    filter_image[i, j, c] = total_val / total_weight
                else:
                    filter_image[i, j, c] = center_val  # Если нет подходящих пикселей, оставляем как есть
        print(f"Обработка: {ceil(100 * (c+1) / channels)}%")
    return np.clip(filter_image, 0, 255).astype(np.uint8)


def absolute_difference_map(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """Карта абсолютной разности между двумя изображениями - оптимизированная версия."""
    diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
    return np.clip(diff, 0, 255).astype(np.uint8)


# резкость

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
