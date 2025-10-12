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
    n, m, channels = image.shape
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge') # добавляем по краям, но не в каналы
    filter_image = np.zeros_like(image)
    for c in range(channels):
        for i in range(n):
            for j in range(m):
                kernel = padded_image[i:i+kernel_size,j:j+kernel_size, c]
                filter_image[i, j, c] = np.average(kernel)
        print(f"Обработка: {ceil(100 * (c+1) / channels)}%")
    return np.clip(filter_image, 0, 255).astype(np.uint8)


def _gaussian_kernel(size: int, sigma: float = 5) -> np.ndarray:
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
    kernel = _gaussian_kernel(kernel_size, sigma)

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


def get_hf_simple(original_image: np.ndarray, lf_image: np.ndarray, c: float) -> np.ndarray:
    high_pass = original_image - c * lf_image
    high_pass = 255* ((high_pass - high_pass.min())/
                      (high_pass.max() - high_pass.min()))
    return high_pass.astype(np.uint8)
    
    
