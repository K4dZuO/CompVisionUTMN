"""
Функции визуализации и сравнения результатов сегментации.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def show_comparison(image: np.ndarray, result: np.ndarray, 
                   hist: np.ndarray = None, title: str = "Comparison") -> Figure:
    """
    Строит сравнение исходного изображения, гистограммы и результата сегментации.
    
    Параметры:
        image: np.ndarray — исходное изображение
        result: np.ndarray — результат сегментации
        hist: np.ndarray — гистограмма (опционально)
        title: str — заголовок
    
    Возвращает:
        Figure — объект фигуры matplotlib
    """
    if hist is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')
        
        axes[1].imshow(result, cmap='gray')
        axes[1].set_title('Результат сегментации')
        axes[1].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')
        
        axes[1].bar(range(len(hist)), hist)
        axes[1].set_title('Гистограмма')
        axes[1].set_xlabel('Интенсивность')
        axes[1].set_ylabel('Частота')
        
        axes[2].imshow(result, cmap='gray')
        axes[2].set_title('Результат сегментации')
        axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def show_multiple_results(image: np.ndarray, results: dict, 
                         hist: np.ndarray = None) -> Figure:
    """
    Показывает несколько результатов сегментации рядом.
    
    Параметры:
        image: np.ndarray — исходное изображение
        results: dict — словарь {название: результат}
        hist: np.ndarray — гистограмма (опционально)
    
    Возвращает:
        Figure — объект фигуры matplotlib
    """
    n_results = len(results)
    cols = 3
    rows = (n_results + cols - 1) // cols + (1 if hist is not None else 0)
    
    fig = plt.figure(figsize=(15, 5 * rows))
    
    # Исходное изображение
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(image, cmap='gray')
    ax.set_title('Исходное изображение')
    ax.axis('off')
    
    # Гистограмма
    if hist is not None:
        ax = plt.subplot(rows, cols, 2)
        ax.bar(range(len(hist)), hist)
        ax.set_title('Гистограмма')
        ax.set_xlabel('Интенсивность')
        ax.set_ylabel('Частота')
    
    # Результаты
    idx = 2 if hist is not None else 1
    for name, result in results.items():
        idx += 1
        ax = plt.subplot(rows, cols, idx)
        ax.imshow(result, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def compute_segmentation_metrics(original: np.ndarray, segmented: np.ndarray) -> dict:
    """
    Вычисляет метрики качества сегментации.
    
    Параметры:
        original: np.ndarray — исходное изображение
        segmented: np.ndarray — результат сегментации
    
    Возвращает:
        dict — словарь с метриками
    """
    # Бинаризация для подсчёта
    binary = (segmented > 127).astype(np.uint8)
    
    # Доля белых пикселей
    white_ratio = np.sum(binary == 1) / binary.size
    
    # Количество сегментов (простая оценка через связные компоненты)
    # Используем упрощённый подсчёт
    num_segments = estimate_connected_components(binary)
    
    # Энтропия
    hist = np.histogram(segmented.flatten(), bins=256, range=(0, 256))[0]
    hist = hist[hist > 0]  # Убираем нули
    probs = hist / hist.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    return {
        'white_ratio': white_ratio,
        'num_segments': num_segments,
        'entropy': entropy
    }


def estimate_connected_components(binary: np.ndarray) -> int:
    """
    Упрощённая оценка количества связных компонент.
    
    Параметры:
        binary: np.ndarray — бинарное изображение (0 и 1)
    
    Возвращает:
        int — приблизительное количество компонент
    """
    # Простая оценка: считаем количество "островков" через подсчёт переходов
    h, w = binary.shape
    transitions = 0
    for i in range(1, h):
        for j in range(1, w):
            if binary[i, j] != binary[i-1, j]:
                transitions += 1
            if binary[i, j] != binary[i, j-1]:
                transitions += 1
    
    return max(1, transitions // 4)

