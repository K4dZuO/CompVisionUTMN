"""
Операторы градиента для выделения краёв.
"""
import numpy as np


def apply_sobel(image: np.ndarray):
    """
    Применяет оператор Собеля для вычисления градиентов.
    Возвращает:
        tuple[Gx, Gy] — градиенты по X и Y
    """
    # Ядра Собеля
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)
    
    return apply_operator(image, sobel_x, sobel_y)


def apply_prewitt(image: np.ndarray):
    """
    Применяет оператор Превитта для вычисления градиентов.
    """
    # Ядра Превитта
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]], dtype=np.float32)
    
    return apply_operator(image, prewitt_x, prewitt_y)


def apply_roberts(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Применяет оператор Робертса для вычисления градиентов.
    """
    # Ядра Робертса (2x2)
    image_float = image.astype(np.float32)
    h, w = image_float.shape
    Gx = np.zeros_like(image_float, dtype=np.float32)
    Gy = np.zeros_like(image_float, dtype=np.float32)
    
    for i in range(h - 1):
        for j in range(w - 1):
            Gx[i, j] = image_float[i, j] - image_float[i + 1, j + 1]
            Gy[i, j] = image_float[i, j + 1] - image_float[i + 1, j]
    
    return Gx, Gy


def apply_operator(image: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Применяет оператор свёртки для вычисления градиентов.
    Возвращает:
        tuple[Gx, Gy] — градиенты по X и Y
    """
    h, w = image.shape
    kh, kw = kernel_x.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Padding для сохранения размера
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    Gx = np.zeros_like(image, dtype=np.float64)
    Gy = np.zeros_like(image, dtype=np.float64)
    
    for i in range(h):
        for j in range(w):
            # Свёртка для Gx
            region = padded[i:i + kh, j:j + kw]
            Gx[i, j] = np.sum(region * kernel_x)
            
            # Свёртка для Gy
            Gy[i, j] = np.sum(region * kernel_y)
    
    return Gx, Gy


def compute_gradient_magnitude(Gx: np.ndarray, Gy: np.ndarray) -> np.ndarray:
    """
    Вычисляет градиента G = sqrt(Gx² + Gy²).
    """
    return np.sqrt(Gx**2 + Gy**2)


def threshold_edges(gradient_magnitude: np.ndarray, T_edge: float) -> np.ndarray:
    """
    Бинаризует градиент по порогу.
    """
    binary = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    binary[gradient_magnitude > T_edge] = 255
    return binary


def edge_segmentation(image: np.ndarray, operator: str = 'sobel', T_edge: float = 50.0) -> np.ndarray:
    """
    Выполняет сегментацию по краям с использованием выбранного оператора.
    """
    # Выбор оператора
    if operator.lower() == 'sobel': # как превитт, но акцент на центральные
        Gx, Gy = apply_sobel(image)
    elif operator.lower() == 'prewitt': # разница между противоположными сторонами
        Gx, Gy = apply_prewitt(image)
    elif operator.lower() == 'roberts': # нет центра, но быстрый
        Gx, Gy = apply_roberts(image)
    else:
        raise ValueError(f"Неизвестный оператор: {operator}")
    
    # Вычисление модуля градиента
    magnitude = compute_gradient_magnitude(Gx, Gy)
    
    # Бинаризация
    result = threshold_edges(magnitude, T_edge)
    
    return result

