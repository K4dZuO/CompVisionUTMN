import numpy as np
from math import ceil
from matplotlib import pyplot as plt
from PIL import Image
import cv2



def get_histogram(image_2d: np.ndarray) -> np.ndarray:
    """Считает гистограмму вручную (с NumPy)."""
    hist, _ = np.histogram(image_2d.ravel(), bins=256, range=(0, 256))
    return hist.astype(np.int32)


def draw_histogram_image(hist: np.ndarray, color: str = 'gray') -> Image.Image:
    """
    Рисует гистограмму с помощью matplotlib и возвращает как PIL.Image.
    :param hist: массив длиной 256 (гистограмма)
    :param title: заголовок графика
    :param color: цвет столбцов
    :return: PIL.Image
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(hist)), hist, width=1, color=color.lower(), edgecolor='none')
    ax.set_title(color.upper())
    ax.set_xlim(0, 255)
    ax.set_ylim(0, hist.max() * 1.1)

    # Убираем лишние отступы
    fig.tight_layout()

    # Сохраняем в BytesIO
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)

    pil_img = Image.open(buf)
    return pil_img

def custom_colormap_pillow(image_np):
    # В серый
    img = Image.fromarray(image_np).convert("L")
    gray = np.array(img)

    # Нормализация к диапазону [0, 255]
    norm = (gray - gray.min()) / (np.ptp(gray) + 1e-5) * 255
    norm = norm.astype(np.uint8)

    # Простая псевдо-палитра (синий->зеленый->красный)
    colored = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    colored[..., 0] = 255 - norm
    colored[..., 1] = np.abs(128 - norm) * 2
    colored[..., 2] = norm

    return colored


def custom_colormap_manual(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    norm = (gray - gray.min()) / (np.ptp(gray) + 1e-5)

    colored = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    colored[..., 0] = (norm * 255).astype(np.uint8)
    colored[..., 1] = ((1 - np.abs(norm - 0.5) * 2) * 255).astype(np.uint8)
    colored[..., 2] = ((1 - norm) * 255).astype(np.uint8)

    return colored

def avarage_filter(image: np.ndarray, kernel_size: int=5) -> np.ndarray:
    """
    Усредняющий фильтр
    ОПТИМИЗИРОВАНО: Использует scipy.ndimage.uniform_filter для ускорения
    
    Args:
        image: входное изображение (H, W, C)
        kernel_size: размер ядра фильтра
    
    Returns:
        отфильтрованное изображение
    """
    from scipy.ndimage import uniform_filter
    
    # ОПТИМИЗАЦИЯ 1: Проверка входных данных
    if image is None:
        raise ValueError("Изображение не может быть None")
    
    if len(image.shape) != 3:
        raise ValueError("Изображение должно быть 3D (H, W, C)")
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        kernel_size = 5  # Используем нечетный размер по умолчанию
    
    # ОПТИМИЗАЦИЯ 2: Использование scipy.ndimage.uniform_filter для ускорения
    # Этот метод намного быстрее циклов и использует оптимизированные алгоритмы
    filter_image = np.zeros_like(image, dtype=np.float32)
    
    for c in range(image.shape[2]):
        # ОПТИМИЗАЦИЯ 3: Векторизованная обработка каждого канала
        filter_image[:, :, c] = uniform_filter(
            image[:, :, c].astype(np.float32), 
            size=kernel_size, 
            mode='nearest'
        )
    
    # ОПТИМИЗАЦИЯ 4: Эффективное обрезание значений
    return np.clip(filter_image, 0, 255).astype(np.uint8)


def _gaussian_kernel(size: int, sigma: float = 5) -> np.ndarray:
    """
    Создаёт гауссово ядро размером size x size.
    ОПТИМИЗИРОВАНО: Использует векторизованные операции numpy
    """
    # ОПТИМИЗАЦИЯ 1: Проверка параметров
    if size <= 0 or size % 2 == 0:
        size = 5
    if sigma <= 0:
        sigma = 1.0
    
    # ОПТИМИЗАЦИЯ 2: Векторизованное создание координатной сетки
    center = size // 2 
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    
    # ОПТИМИЗАЦИЯ 3: Векторизованное вычисление гауссова ядра
    s = 2 * (sigma ** 2)
    kernel = np.exp(-(x**2 + y**2) / s)
    
    # ОПТИМИЗАЦИЯ 4: Векторизованная нормализация
    kernel = kernel / np.sum(kernel)
    
    return kernel.astype(np.float32)  # Используем float32 для экономии памяти

def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Гауссов фильтр
    ОПТИМИЗИРОВАНО: Использует scipy.ndimage.gaussian_filter для ускорения
    
    Args:
        image: входное изображение (H, W, C)
        sigma: стандартное отклонение гауссова ядра
    
    Returns:
        отфильтрованное изображение
    """
    from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    
    # ОПТИМИЗАЦИЯ 1: Проверка входных данных
    if image is None:
        raise ValueError("Изображение не может быть None")
    
    if len(image.shape) != 3:
        raise ValueError("Изображение должно быть 3D (H, W, C)")
    
    if sigma <= 0:
        sigma = 1.0
    
    # ОПТИМИЗАЦИЯ 2: Использование scipy.ndimage.gaussian_filter
    # Этот метод намного быстрее ручной реализации и использует оптимизированные алгоритмы
    filter_image = np.zeros_like(image, dtype=np.float32)
    
    for c in range(image.shape[2]):
        # ОПТИМИЗАЦИЯ 3: Векторизованная обработка каждого канала
        filter_image[:, :, c] = scipy_gaussian_filter(
            image[:, :, c].astype(np.float32), 
            sigma=sigma, 
            mode='nearest'
        )
    
    # ОПТИМИЗАЦИЯ 4: Эффективное обрезание значений
    return np.clip(filter_image, 0, 255).astype(np.uint8)


def get_hf_simple(original_image: np.ndarray, lf_image: np.ndarray, c: float) -> np.ndarray:
    """
    Получает высокочастотное изображение по правилу ВЧ = ИСХ - РАЗМ*c
    Решает проблему отрицательной яркости через нормализацию
    """
    high_pass = original_image.astype(np.float32) - c * lf_image.astype(np.float32)
    
    # Решение проблемы отрицательной яркости - нормализация к [0, 255]
    high_pass_min = high_pass.min()
    high_pass_max = high_pass.max()
    
    if high_pass_max > high_pass_min:  # избегаем деления на ноль
        high_pass = 255 * (high_pass - high_pass_min) / (high_pass_max - high_pass_min)
    else:
        high_pass = np.zeros_like(high_pass)
    
    return np.clip(high_pass, 0, 255).astype(np.uint8)


def apply_convolution_filter(image: np.ndarray, kernel: np.ndarray, normalize: bool = True, add_128: bool = False) -> np.ndarray:
    """
    Применяет фильтр через операцию свёртки с матрицей произвольного размера
    ОПТИМИЗИРОВАНО: Использует scipy.signal.convolve2d для ускорения
    
    Args:
        image: входное изображение (H, W, C)
        kernel: матрица фильтра (n, n)
        normalize: нормализация (деление на сумму элементов матрицы)
        add_128: прибавление 128 к результату
    
    Returns:
        отфильтрованное изображение
    """
    from scipy import signal
    
    # ОПТИМИЗАЦИЯ 1: Проверка входных данных для предотвращения ошибок
    if image is None or kernel is None:
        raise ValueError("Изображение и ядро не могут быть None")
    
    if len(kernel.shape) != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Ядро должно быть квадратной матрицей")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Изображение должно быть 2D или 3D")
    
    # ОПТИМИЗАЦИЯ 2: Нормализация ядра заранее для экономии вычислений
    if normalize:
        kernel_sum = np.sum(kernel)
        if abs(kernel_sum) < 1e-10:  # Избегаем деления на ноль
            kernel = kernel.copy()
        else:
            kernel = kernel / kernel_sum
    
    # ОПТИМИЗАЦИЯ 3: Обработка 2D изображений без создания лишних измерений
    if len(image.shape) == 2:
        # Для 2D изображений используем прямое применение свёртки
        result = signal.convolve2d(image.astype(np.float32), kernel, mode='same', boundary='symm')
    else:
        # ОПТИМИЗАЦИЯ 4: Векторизованная обработка всех каналов одновременно
        h, w, c = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        
        for ch in range(c):
            # Используем scipy.signal.convolve2d вместо циклов
            result[:, :, ch] = signal.convolve2d(
                image[:, :, ch].astype(np.float32), 
                kernel, 
                mode='same', 
                boundary='symm'
            )
    
    # ОПТИМИЗАЦИЯ 5: Векторизованное прибавление 128
    if add_128:
        result += 128
    
    # ОПТИМИЗАЦИЯ 6: Эффективное обрезание значений
    result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def parse_kernel_from_string(kernel_str: str, size: int) -> np.ndarray:
    """
    Парсит строку с элементами матрицы в numpy массив
    
    Args:
        kernel_str: строка с элементами, разделенными пробелами, запятыми или переносами
        size: размер матрицы (n x n)
    
    Returns:
        матрица numpy размера (size, size)
    """
    # Заменяем запятые на пробелы и разделяем
    elements = kernel_str.replace(',', ' ').split()
    
    # Преобразуем в числа
    try:
        values = [float(x) for x in elements]
    except ValueError:
        raise ValueError("Не удалось преобразовать элементы матрицы в числа")
    
    # Проверяем количество элементов
    expected_size = size * size
    if len(values) != expected_size:
        raise ValueError(f"Ожидается {expected_size} элементов, получено {len(values)}")
    
    return np.array(values).reshape(size, size)


def get_standard_kernels():
    """
    Возвращает словарь со стандартными матрицами фильтров
    """
    kernels = {
        "Размытие 3x3": np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]) / 9,
        
        "Размытие 5x5": np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]) / 25,
        
        "Резкость": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),
        
        "Собель X": np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        
        "Собель Y": np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        
        "Лапласиан": np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]),
        
        "Лапласиан диагональный": np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]),
        
        "Превитт X": np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]),
        
        "Превитт Y": np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ])
    }
    
    return kernels


def harris_corner_detection(image: np.ndarray, k: float = 0.04, threshold: float = 0.01) -> np.ndarray:
    """
    Детекция углов методом Харриса
    ОПТИМИЗИРОВАНО: Улучшена обработка градиентов и нормализация
    
    Args:
        image: входное изображение (H, W, C) или (H, W)
        k: параметр Харриса (обычно 0.04-0.06)
        threshold: порог для отбора углов
    
    Returns:
        изображение с отмеченными углами
    """
    # ОПТИМИЗАЦИЯ 1: Проверка входных данных
    if image is None:
        raise ValueError("Изображение не может быть None")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Изображение должно быть 2D или 3D")
    
    # ОПТИМИЗАЦИЯ 2: Эффективное преобразование в серый
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # ОПТИМИЗАЦИЯ 3: Использование float32 для экономии памяти
    gray = gray.astype(np.float32)
    
    # ОПТИМИЗАЦИЯ 4: Вычисление градиентов с правильным типом данных
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # ОПТИМИЗАЦИЯ 5: Векторизованное вычисление элементов матрицы Харриса
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # ОПТИМИЗАЦИЯ 6: Использование большего размера ядра для лучшего сглаживания
    kernel_size = 5  # Увеличиваем размер ядра для лучшего качества
    sigma = 1.0
    Ixx = cv2.GaussianBlur(Ixx, (kernel_size, kernel_size), sigma)
    Ixy = cv2.GaussianBlur(Ixy, (kernel_size, kernel_size), sigma)
    Iyy = cv2.GaussianBlur(Iyy, (kernel_size, kernel_size), sigma)
    
    # ОПТИМИЗАЦИЯ 7: Векторизованное вычисление детерминанта и следа
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    
    # ОПТИМИЗАЦИЯ 8: Вычисление отклика Харриса с проверкой на деление на ноль
    harris_response = det - k * (trace * trace)
    
    # ОПТИМИЗАЦИЯ 9: Улучшенная нормализация с проверкой на максимум
    harris_response = np.maximum(harris_response, 0)
    max_val = harris_response.max()
    if max_val > 0:
        harris_response = harris_response / max_val
    
    # ОПТИМИЗАЦИЯ 10: Использование non-maximum suppression для лучшего качества
    # Находим локальные максимумы
    from scipy.ndimage import maximum_filter
    local_maxima = maximum_filter(harris_response, size=3) == harris_response
    corners = (harris_response > threshold) & local_maxima
    
    # Создаем результат
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    # ОПТИМИЗАЦИЯ 11: Векторизованное отмечение углов
    result[corners] = [255, 0, 0]
    
    return result


def shi_tomasi_corner_detection(image: np.ndarray, max_corners: int = 100, quality_level: float = 0.01, min_distance: int = 10) -> np.ndarray:
    """
    Детекция углов методом Shi-Tomasi
    ОПТИМИЗИРОВАНО: Исправлена ошибка numpy и улучшена обработка углов
    
    Args:
        image: входное изображение (H, W, C) или (H, W)
        max_corners: максимальное количество углов
        quality_level: минимальное качество угла (0-1)
        min_distance: минимальное расстояние между углами
    
    Returns:
        изображение с отмеченными углами
    """
    # ОПТИМИЗАЦИЯ 1: Проверка входных данных
    if image is None:
        raise ValueError("Изображение не может быть None")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Изображение должно быть 2D или 3D")
    
    # ОПТИМИЗАЦИЯ 2: Проверка параметров
    if max_corners <= 0:
        max_corners = 100
    if quality_level <= 0 or quality_level >= 1:
        quality_level = 0.01
    if min_distance <= 0:
        min_distance = 10
    
    # ОПТИМИЗАЦИЯ 3: Эффективное преобразование в серый
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # ОПТИМИЗАЦИЯ 4: Проверка размера изображения
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        # Слишком маленькое изображение для детекции углов
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result
    
    # ОПТИМИЗАЦИЯ 5: Использование float32 для лучшей точности
    gray = gray.astype(np.float32)
    
    try:
        # ОПТИМИЗАЦИЯ 6: Находим углы с обработкой ошибок
        corners = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=max_corners, 
            qualityLevel=quality_level, 
            minDistance=min_distance,
            useHarrisDetector=False,  # Используем именно Shi-Tomasi
            k=0.04
        )
    except cv2.error as e:
        print(f"Ошибка OpenCV в Shi-Tomasi: {e}")
        corners = None
    
    # Создаем результат
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    # ОПТИМИЗАЦИЯ 7: Безопасная обработка углов
    if corners is not None and len(corners) > 0:
        # ОПТИМИЗАЦИЯ 8: Векторизованное отмечение углов
        corners_int = corners.astype(np.int32)
        
        # Проверяем границы изображения
        h, w = result.shape[:2]
        valid_corners = (
            (corners_int[:, 0, 0] >= 0) & 
            (corners_int[:, 0, 0] < w) & 
            (corners_int[:, 0, 1] >= 0) & 
            (corners_int[:, 0, 1] < h)
        )
        
        if np.any(valid_corners):
            valid_corners_int = corners_int[valid_corners]
            
            # ОПТИМИЗАЦИЯ 9: Использование cv2.circle для каждого угла
            for corner in valid_corners_int:
                x, y = corner[0]
                cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    return result


def sobel_edge_detection(image: np.ndarray, add_128: bool = True) -> np.ndarray:
    """
    Детекция границ оператором Собеля
    ОПТИМИЗИРОВАНО: Улучшена обработка градиентов и нормализация
    
    Args:
        image: входное изображение (H, W, C) или (H, W)
        add_128: прибавлять ли 128 к результату
    
    Returns:
        изображение с выделенными границами
    """
    # ОПТИМИЗАЦИЯ 1: Проверка входных данных
    if image is None:
        raise ValueError("Изображение не может быть None")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Изображение должно быть 2D или 3D")
    
    # ОПТИМИЗАЦИЯ 2: Эффективное преобразование в серый
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # ОПТИМИЗАЦИЯ 3: Использование float32 для экономии памяти
    gray = gray.astype(np.float32)
    
    # ОПТИМИЗАЦИЯ 4: Вычисление градиентов с правильным типом данных
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # ОПТИМИЗАЦИЯ 5: Векторизованное вычисление магнитуды градиента
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # ОПТИМИЗАЦИЯ 6: Безопасная нормализация с проверкой на максимум
    max_magnitude = magnitude.max()
    if max_magnitude > 0:
        magnitude = magnitude / max_magnitude * 255
    
    # ОПТИМИЗАЦИЯ 7: Векторизованное прибавление 128
    if add_128:
        magnitude += 128
    
    # ОПТИМИЗАЦИЯ 8: Эффективное обрезание значений
    magnitude = np.clip(magnitude, 0, 255)
    
    # ОПТИМИЗАЦИЯ 9: Эффективное создание RGB изображения
    result = np.stack([magnitude, magnitude, magnitude], axis=2)
    
    return result.astype(np.uint8)


def canny_edge_detection(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Детекция границ алгоритмом Канни
    ОПТИМИЗИРОВАНО: Добавлена проверка параметров и обработка ошибок
    
    Args:
        image: входное изображение (H, W, C) или (H, W)
        low_threshold: нижний порог
        high_threshold: верхний порог
    
    Returns:
        изображение с выделенными границами
    """
    # ОПТИМИЗАЦИЯ 1: Проверка входных данных
    if image is None:
        raise ValueError("Изображение не может быть None")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Изображение должно быть 2D или 3D")
    
    # ОПТИМИЗАЦИЯ 2: Проверка и корректировка порогов
    if low_threshold < 0:
        low_threshold = 50
    if high_threshold < 0:
        high_threshold = 150
    if low_threshold >= high_threshold:
        high_threshold = low_threshold + 50
    
    # ОПТИМИЗАЦИЯ 3: Эффективное преобразование в серый
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # ОПТИМИЗАЦИЯ 4: Проверка размера изображения
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        # Слишком маленькое изображение для Canny
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result
    
    try:
        # ОПТИМИЗАЦИЯ 5: Применяем алгоритм Канни с обработкой ошибок
        edges = cv2.Canny(gray, low_threshold, high_threshold)
    except cv2.error as e:
        print(f"Ошибка OpenCV в Canny: {e}")
        # Возвращаем исходное изображение в случае ошибки
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        return result
    
    # ОПТИМИЗАЦИЯ 6: Эффективное создание RGB изображения
    result = np.stack([edges, edges, edges], axis=2)
    
    return result
    
    
