from typing import List
from PIL import Image
import numpy as np


def get_histogram(image_2d: np.ndarray) -> np.ndarray:
    """
    Считает гистограмму вручную (с NumPy).
    :param image_2d: 2D numpy array (grayscale или один цветовой канал)
    :return: массив длиной 256 (гистограмма)
    """
    # Используем np.bincount — быстрее, чем цикл
    hist, _ = np.histogram(image_2d.ravel(), bins=256, range=(0, 256))
    return hist.astype(np.int32)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Без GUI-бэкенда
from matplotlib import pyplot as plt
from PIL import Image
from typing import Tuple


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

