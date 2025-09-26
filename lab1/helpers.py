import numpy as np
import matplotlib
matplotlib.use('Agg')  # Без GUI-бэкенда
from matplotlib import pyplot as plt
from PIL import Image
from typing import Tuple
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
    norm = (gray - gray.min()) / (gray.ptp() + 1e-5) * 255
    norm = norm.astype(np.uint8)

    # Простая псевдо-палитра (синий->зеленый->красный)
    colored = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    colored[..., 0] = 255 - norm            # красный канал уменьшается
    colored[..., 1] = np.abs(128 - norm) * 2  # зелёный от серого
    colored[..., 2] = norm                  # синий прямо из нормализованного

    return colored
    
def custom_colormap_manual(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    norm = (gray - gray.min()) / (gray.ptp() + 1e-5)

    colored = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    # Красный = высокий уровень
    colored[..., 0] = (norm * 255).astype(np.uint8)
    # Зелёный = средние значения
    colored[..., 1] = ((1 - np.abs(norm - 0.5) * 2) * 255).astype(np.uint8)
    # Синий = обратный уровень
    colored[..., 2] = ((1 - norm) * 255).astype(np.uint8)

    return colored