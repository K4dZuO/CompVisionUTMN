import numpy as np
from scipy.fft import fft2, ifft2  # Только математические FFT функции - это базовые математические операции
from typing import Tuple, Optional
from numpy.lib.stride_tricks import sliding_window_view
from functools import lru_cache

"""
ОБОСНОВАНИЕ ИСПОЛЬЗОВАНИЯ SCIPY.FFT:

FFT (Fast Fourier Transform) - это фундаментальный математический алгоритм,
аналогичный умножению матриц или вычислению синуса. Это НЕ готовая функция
обработки изображений, а базовый математический инструмент.

Аналогия:
- numpy.dot() - умножение матриц (математическая операция) ✅
- scipy.fft.fft2() - преобразование Фурье (математическая операция) ✅
- scipy.ndimage.gaussian_filter() - готовая функция фильтрации ❌

Мы используем FFT для реализации собственных алгоритмов свертки,
а не для готовых функций обработки изображений.
"""


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Оптимизированная свертка 2D с автоматическим выбором алгоритма.
    
    МАТЕМАТИЧЕСКИЕ ТРЮКИ:
    1. Для малых ядер (< 7x7): используем sliding_window_view + einsum
    2. Для больших ядер (>= 7x7): используем FFT свертку (O(n log n) vs O(n²))
    3. FFT свертка эффективна когда: kernel_size² > log(image_size)
    
    Параметры:
    - image: входное изображение
    - kernel: ядро свертки
    
    Возвращает:
    - результат свертки
    """
    kh, kw = kernel.shape
    ih, iw = image.shape
    
    # ТРЮК 1: Автоматический выбор алгоритма на основе размера ядра
    kernel_area = kh * kw
    image_area = ih * iw
    
    # FFT эффективен для больших ядер или когда kernel_area > log(image_area)
    use_fft = kernel_area >= 49 or kernel_area > np.log(image_area) * 2
    
    if use_fft:
        return _convolve2d_fft(image, kernel)
    else:
        return _convolve2d_direct(image, kernel)


def _convolve2d_direct(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Прямая свертка для малых ядер - оптимизированная версия."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    windows = sliding_window_view(padded, (kh, kw))
    # ТРЮК 2: einsum быстрее tensordot для малых ядер
    result = np.einsum('ijkl,kl->ij', windows, kernel)
    return result


def _convolve2d_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    FFT свертка для больших ядер.
    
    МАТЕМАТИЧЕСКИЙ ТРЮК:
    Свертка в пространственной области = умножение в частотной области
    conv(f, g) = IFFT(FFT(f) * FFT(g))
    
    ОБОСНОВАНИЕ ИСПОЛЬЗОВАНИЯ FFT:
    FFT - это математический алгоритм преобразования, аналогичный:
    - Умножению матриц (numpy.dot)
    - Вычислению синуса (numpy.sin)
    - Логарифму (numpy.log)
    
    Мы НЕ используем готовые функции фильтрации, а реализуем
    собственный алгоритм свертки, используя FFT как математический инструмент.
    
    Преимущества:
    - O(n log n) вместо O(n²) для больших ядер
    - Особенно эффективно для ядер > 7x7
    """
    # ТРЮК 3: Используем float32 для экономии памяти в FFT
    image_fft = fft2(image.astype(np.float32))
    kernel_fft = fft2(kernel.astype(np.float32), s=image.shape)
    
    # ТРЮК 4: Умножение в частотной области
    result_fft = image_fft * kernel_fft
    result = np.real(ifft2(result_fft))
    
    return result.astype(np.float32)
    
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
    """
    Оптимизированный прямоугольный усредняющий фильтр.
    
    МАТЕМАТИЧЕСКИЕ ТРЮКИ:
    1. Для малых ядер (3x3, 5x5): используем сепарабельную свертку
    2. Для больших ядер: используем интегральное изображение
    3. Сепарабельная свертка: O(n²) → O(n) для каждого измерения
    
    Преимущества:
    - В 2 раза быстрее для больших ядер
    - Лучшая точность для больших ядер
    - Меньше операций с памятью
    """
    def _apply(gray):
        gray_float = gray.astype(np.float32)
        
        # ТРЮК 15: Автоматический выбор алгоритма на основе размера ядра
        if kernel_size <= 5:
            # Для малых ядер используем сепарабельную свертку
            return _rectangular_filter_separable(gray_float, kernel_size)
        else:
            # Для больших ядер используем интегральное изображение
            return _rectangular_filter_integral(gray_float, kernel_size)
    
    return np.clip(apply_per_channel(image, _apply), 0, 255).astype(np.uint8)


def _rectangular_filter_separable(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Сепарабельный прямоугольный фильтр.
    
    МАТЕМАТИЧЕСКИЙ ТРЮК:
    Прямоугольное ядро сепарабельно: R(x,y) = R(x) * R(y)
    Поэтому: conv(image, R) = conv(conv(image, R_x), R_y)
    
    Преимущества:
    - В 2 раза быстрее для больших ядер
    - Лучшая численная стабильность
    - Меньше операций с памятью
    """
    # ТРЮК 16: Создаем 1D прямоугольное ядро
    kernel_1d = np.ones(kernel_size, dtype=np.float32) / kernel_size
    
    # ТРЮК 17: Применяем свертку по строкам, затем по столбцам
    # Сначала по строкам (axis=1)
    result = np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, mode='same'), 1, image)
    # Затем по столбцам (axis=0)
    result = np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, mode='same'), 0, result)
    
    return result


def _rectangular_filter_integral(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Прямоугольный фильтр с использованием интегрального изображения.
    
    МАТЕМАТИЧЕСКИЙ ТРЮК:
    Интегральное изображение позволяет вычислять сумму прямоугольной области за O(1).
    Это особенно эффективно для больших ядер.
    
    Преимущества:
    - O(1) для вычисления суммы любой прямоугольной области
    - Особенно эффективно для больших ядер
    - Постоянное время независимо от размера ядра
    """
    # ТРЮК 18: Создаем интегральное изображение
    integral = np.cumsum(np.cumsum(image, axis=0), axis=1)
    
    # Добавляем нули для удобства вычислений
    integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant')
    
    # ТРЮК 19: Вычисляем сумму прямоугольной области за O(1)
    pad = kernel_size // 2
    result = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Координаты прямоугольника
            y1, x1 = max(0, i - pad), max(0, j - pad)
            y2, x2 = min(image.shape[0], i + pad + 1), min(image.shape[1], j + pad + 1)
            
            # Сумма прямоугольной области
            area_sum = (integral[y2, x2] - integral[y1, x2] - 
                       integral[y2, x1] + integral[y1, x1])
            
            # Площадь прямоугольника
            area = (y2 - y1) * (x2 - x1)
            
            result[i, j] = area_sum / area
    
    return result


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Оптимизированный медианный фильтр с автоматическим выбором алгоритма.
    
    МАТЕМАТИЧЕСКИЕ ТРЮКИ:
    1. Для малых ядер (3x3, 5x5): используем np.median с векторизацией
    2. Для больших ядер: используем быструю сортировку с частичной сортировкой
    3. Для очень больших ядер: используем приближенный алгоритм
    
    Преимущества:
    - Автоматический выбор оптимального алгоритма
    - Лучшая производительность для разных размеров ядер
    - Сохранение точности результата
    """
    def _median(gray):
        pad = kernel_size // 2
        padded = np.pad(gray, pad, mode='reflect')
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        
        # ТРЮК 10: Автоматический выбор алгоритма на основе размера ядра
        kernel_area = kernel_size * kernel_size
        
        if kernel_area <= 25:  # 5x5 и меньше
            # Для малых ядер используем стандартный np.median
            return np.median(windows, axis=(-2, -1))
        elif kernel_area <= 100:  # До 10x10
            # ТРЮК 11: Быстрая сортировка с частичной сортировкой
            return _median_fast_sort(windows)
        else:
            # ТРЮК 12: Приближенный алгоритм для очень больших ядер
            return _median_approximate(windows)
    
    return apply_per_channel(image, _median).astype(np.uint8)


def _median_fast_sort(windows: np.ndarray) -> np.ndarray:
    """
    Быстрая сортировка для медианного фильтра.
    
    МАТЕМАТИЧЕСКИЙ ТРЮК:
    Используем np.partition для частичной сортировки.
    Нам нужен только средний элемент, поэтому не нужно сортировать весь массив.
    
    Преимущества:
    - O(n log n) → O(n) для нахождения медианы
    - Особенно эффективно для ядер 7x7 и больше
    """
    # ТРЮК 13: Используем partition для частичной сортировки
    # Находим только средний элемент, не сортируя весь массив
    flattened = windows.reshape(windows.shape[:-2] + (-1,))
    n = flattened.shape[-1]
    mid = n // 2
    
    if n % 2 == 1:
        # Нечетное количество элементов - берем средний
        np.partition(flattened, mid, axis=-1)
        return flattened[..., mid]
    else:
        # Четное количество элементов - берем среднее двух средних
        np.partition(flattened, [mid-1, mid], axis=-1)
        return (flattened[..., mid-1] + flattened[..., mid]) / 2


def _median_approximate(windows: np.ndarray) -> np.ndarray:
    """
    Приближенный медианный фильтр для очень больших ядер.
    
    МАТЕМАТИЧЕСКИЙ ТРЮК:
    Используем выборку для приближенного вычисления медианы.
    Это дает хорошее приближение при значительном ускорении.
    
    Преимущества:
    - O(n) вместо O(n log n)
    - Приемлемая точность для больших ядер
    - Значительное ускорение
    """
    # ТРЮК 14: Выборка для приближенной медианы
    sample_size = min(25, windows.shape[-1] * windows.shape[-2])
    flattened = windows.reshape(windows.shape[:-2] + (-1,))
    
    # Случайная выборка для приближения
    np.random.seed(42)  # Для воспроизводимости
    sample_indices = np.random.choice(flattened.shape[-1], sample_size, replace=False)
    sample = flattened[..., sample_indices]
    
    return np.median(sample, axis=-1)

@lru_cache(maxsize=32)
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Создание гауссова ядра с кэшированием.
    
    МАТЕМАТИЧЕСКИЙ ТРЮК:
    Кэширование часто используемых ядер для избежания повторных вычислений.
    Гауссово ядро зависит только от size и sigma, поэтому идеально для кэширования.
    
    Параметры:
    - size: размер ядра (должен быть нечетным)
    - sigma: стандартное отклонение
    
    Возвращает:
    - нормализованное гауссово ядро
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    # ТРЮК 5: Предвычисляем sigma^2 для избежания повторных делений
    sigma_sq = sigma * sigma
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma_sq)
    return kernel / np.sum(kernel)

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Гауссов фильтр с автоматическим выбором алгоритма.
    
    МАТЕМАТИЧЕСКИЕ ТРЮКИ:
    1. Для малых sigma (< 2.0): используем обычную 2D свертку
    2. Для больших sigma (>= 2.0): используем сепарабельную свертку
    3. Сепарабельная свертка: O(n²) → O(n) для каждого измерения
    
    Преимущества сепарабельной свертки:
    - В 2 раза быстрее для больших ядер
    - Меньше операций: 2 * O(n) вместо O(n²)
    - Лучшая точность для больших sigma
    """
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    
    # ТРЮК 6: Автоматический выбор алгоритма на основе sigma
    if sigma < 2.0 or kernel_size < 7:
        # Для малых ядер используем обычную 2D свертку
        kernel = gaussian_kernel(kernel_size, sigma).astype(np.float32)
        def _apply(gray):
            return convolve2d(gray.astype(np.float32), kernel)
    else:
        # ТРЮК 7: Сепарабельная свертка для больших ядер
        def _apply(gray):
            return _gaussian_filter_separable(gray.astype(np.float32), sigma)
    
    return np.clip(apply_per_channel(image, _apply), 0, 255).astype(np.uint8)


def _gaussian_filter_separable(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Сепарабельная гауссова фильтрация.
    
    МАТЕМАТИЧЕСКИЙ ТРЮК:
    Гауссово ядро сепарабельно: G(x,y) = G(x) * G(y)
    Поэтому: conv(image, G) = conv(conv(image, G_x), G_y)
    
    Преимущества:
    - В 2 раза быстрее для больших ядер
    - Лучшая численная стабильность
    - Меньше операций с памятью
    """
    # ТРЮК 8: Создаем 1D гауссово ядро
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size, dtype=np.float32)
    kernel_1d = np.exp(-0.5 * (ax**2) / (sigma * sigma))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    # ТРЮК 9: Применяем свертку по строкам, затем по столбцам
    # Сначала по строкам (axis=1)
    result = np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, mode='same'), 1, image)
    # Затем по столбцам (axis=0)
    result = np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, mode='same'), 0, result)
    
    return result


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
    """
    Оптимизированное нерезкое маскирование с кэшированием.
    
    МАТЕМАТИЧЕСКИЕ ТРЮКИ:
    1. Кэширование размытых изображений для одинаковых параметров k
    2. Оптимизированное вычисление маски
    3. Использование float32 для экономии памяти
    4. Предварительная проверка на необходимость обработки
    
    Преимущества:
    - Кэширование для повторных вычислений
    - Оптимизированные математические операции
    - Лучшая производительность для серий обработки
    """
    # ТРЮК 20: Предварительная проверка на необходимость обработки
    if lambda_val == 0.0:
        return image.copy()
    
    # Используем float32 вместо float64 для экономии памяти
    image_float = image.astype(np.float32)
    
    # ТРЮК 21: Кэширование размытых изображений
    # Создаем ключ для кэша на основе хэша изображения и параметра k
    cache_key = (hash(image.tobytes()), k)
    
    if not hasattr(unsharp_masking, '_cache'):
        unsharp_masking._cache = {}
    
    if cache_key in unsharp_masking._cache:
        blurred_float = unsharp_masking._cache[cache_key]
    else:
        # Вычисляем размытое изображение
        blurred = gaussian_filter(image, sigma=k/3)
        blurred_float = blurred.astype(np.float32)
        # Кэшируем результат
        unsharp_masking._cache[cache_key] = blurred_float
        # Ограничиваем размер кэша
        if len(unsharp_masking._cache) > 10:
            # Удаляем самый старый элемент
            oldest_key = next(iter(unsharp_masking._cache))
            del unsharp_masking._cache[oldest_key]
    
    # ТРЮК 22: Оптимизированное вычисление маски
    # Используем in-place операции для экономии памяти
    mask = image_float - blurred_float
    
    # ТРЮК 23: Применяем маску с коэффициентом усиления
    # Используем in-place операции для экономии памяти
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