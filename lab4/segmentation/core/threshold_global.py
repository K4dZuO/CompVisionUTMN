"""
Глобальная пороговая сегментация.
"""
import numpy as np


def threshold_ptile(image: np.ndarray, P: float = 30.0):
    """
    Метод площади (P-tile): вычисляет порог так, чтобы заданный процент пикселей оказался выше порога.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        P: float — процент пикселей, которые должны быть выше порога (0-100)
    
    Возвращает:
        tuple[T, binary_mask] — порог и бинарная маска (0 и 255)
    """
    # Вычисляем порог
    flat = image.flatten()
    sorted_values = np.sort(flat)
    n = len(sorted_values)
    idx = int(n * (1 - P / 100.0))
    idx = max(0, min(idx, n - 1))
    T = sorted_values[idx]
    
    # Бинаризация
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > T] = 255
    
    return T, binary


def threshold_iterative(image: np.ndarray, eps: float = 0.5, max_iter: int = 50):
    """
    Глобальная пороговая сегментация методом последовательных приближений.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        eps: float — критерий останова
        max_iter: int — максимум итераций
    
    Возвращает:
        tuple[T, binary_mask] — порог и бинарная маска (0 и 255)
    """
    # Конвертируем в float для избежания overflow
    image_float = image.astype(np.float64)
    
    # Начальное приближение
    T_old = (np.min(image_float) + np.max(image_float)) / 2.0
    T_new = T_old  # Инициализируем для случая, если цикл не выполнится
    
    for iteration in range(max_iter):
        # Разделяем пиксели на два класса
        class1 = image_float[image_float <= T_old]
        class2 = image_float[image_float > T_old]
        
        if len(class1) == 0 or len(class2) == 0:
            T_new = T_old
            break
        
        # Вычисляем средние
        mu1 = np.mean(class1)
        mu2 = np.mean(class2)
        
        # Новый порог
        T_new = (mu1 + mu2) / 2.0
        
        # Проверка останова
        if abs(T_new - T_old) < eps:
            break
        
        T_old = T_new
    
    T = T_new
    
    # Бинаризация
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image > T] = 255
    
    return T, binary


def kmeans_1d(data: np.ndarray, k: int = 2, max_iter: int = 100, eps: float = 0.1):
    """
    K-средних для 1D-кластеризации значений яркости.
    
    Параметры:
        data: np.ndarray — одномерный массив значений яркости
        k: int — количество кластеров
        max_iter: int — максимум итераций
        eps: float — критерий останова
    
    Возвращает:
        tuple: (centroids, labels) — центроиды и метки кластеров
    """
    n = len(data)
    
    # Инициализация центроидов (равномерно распределённые)
    min_val, max_val = np.min(data), np.max(data)
    centroids = np.linspace(min_val, max_val, k)
    
    labels = np.zeros(n, dtype=np.int32)
    
    for iteration in range(max_iter):
        # Присваивание меток
        for i in range(n):
            distances = np.abs(data[i] - centroids)
            labels[i] = np.argmin(distances)
        
        # Обновление центроидов
        centroids_new = np.zeros(k)
        for j in range(k):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                centroids_new[j] = np.mean(cluster_points)
            else:
                centroids_new[j] = centroids[j]
        
        # Проверка останова
        if np.all(np.abs(centroids_new - centroids) < eps):
            break
        
        centroids = centroids_new
    
    return centroids, labels


def threshold_kmeans(image: np.ndarray, k: int = 2):
    """
    Глобальная пороговая сегментация методом K-средних.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        k: int — количество кластеров
    
    Возвращает:
        tuple[T, binary_mask] — порог и бинарная маска (0 и 255)
    """
    flat = image.flatten().astype(np.float64)
    centroids, labels = kmeans_1d(flat, k=k)
    
    # Сортируем центроиды
    sorted_indices = np.argsort(centroids)
    sorted_centroids = centroids[sorted_indices]
    
    # Для бинаризации используем средний порог между кластерами
    if k == 2:
        T = sorted_centroids[0] + (sorted_centroids[1] - sorted_centroids[0]) / 2.0
    else:
        # Для k > 2 берём порог между первыми двумя кластерами
        T = sorted_centroids[0] + (sorted_centroids[1] - sorted_centroids[0]) / 2.0
    
    # Бинаризация
    binary = np.zeros_like(image, dtype=np.uint8)
    # Присваиваем классам значения на основе центроидов
    for i in range(k):
        orig_idx = sorted_indices[i]
        mask = (labels == orig_idx).reshape(image.shape)
        if i >= k // 2:  # Верхние классы
            binary[mask] = 255
    
    return T, binary


def threshold_multilevel(image: np.ndarray, k: int = 3):
    """
    Многоуровневая пороговая сегментация методом K-средних.
    
    Параметры:
        image: np.ndarray — входное изображение в оттенках серого [0,255]
        k: int — количество кластеров
    
    Возвращает:
        tuple: (thresholds, segmented) — пороги и результат сегментации
    """
    flat = image.flatten().astype(np.float64)
    centroids, labels = kmeans_1d(flat, k=k)
    
    # Сортируем центроиды
    sorted_indices = np.argsort(centroids)
    sorted_centroids = centroids[sorted_indices]
    
    # Вычисляем пороги между кластерами
    thresholds = []
    for i in range(len(sorted_centroids) - 1):
        T = sorted_centroids[i] + (sorted_centroids[i + 1] - sorted_centroids[i]) / 2.0
        thresholds.append(T)
    thresholds = np.array(thresholds)
    
    # Создаём сегментированное изображение
    segmented = np.zeros_like(image, dtype=np.uint8)
    labels_reshaped = labels.reshape(image.shape)
    
    for i in range(k):
        orig_idx = sorted_indices[i]
        mask = (labels_reshaped == orig_idx)
        # Присваиваем значения пропорционально номеру кластера
        value = int(255 * i / (k - 1)) if k > 1 else 0
        segmented[mask] = value
    
    return thresholds, segmented

