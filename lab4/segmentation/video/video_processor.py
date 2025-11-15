"""
Хелпер для обработки видео с использованием готовых методов OpenCV.
Использует готовые функции сегментации OpenCV для покадровой обработки видео.
"""
import cv2
import numpy as np
from typing import Optional, Tuple


def process_video_frame_edges_opencv(frame: np.ndarray, operator: str = 'canny', 
                                     threshold1: float = 50.0, threshold2: float = 150.0,
                                     **kwargs) -> np.ndarray:
    """
    Обрабатывает кадр видео для выделения краёв используя готовые методы OpenCV.
    
    Параметры:
        frame: np.ndarray — кадр видео в оттенках серого [0,255]
        operator: str — оператор ('canny', 'sobel', 'laplacian')
        threshold1: float — первый порог для Canny
        threshold2: float — второй порог для Canny
        **kwargs — дополнительные параметры для операторов
    
    Возвращает:
        np.ndarray — бинарная маска краёв (0 и 255)
    """
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    if operator.lower() == 'canny':
        # Используем готовый алгоритм Canny из OpenCV
        edges = cv2.Canny(gray_frame, int(threshold1), int(threshold2))
        return edges
    elif operator.lower() == 'sobel':
        # Используем Sobel из OpenCV
        sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # Бинаризация по порогу
        threshold = kwargs.get('threshold', threshold1)
        binary = np.zeros_like(gray_frame, dtype=np.uint8)
        binary[magnitude > threshold] = 255
        return binary
    elif operator.lower() == 'laplacian':
        # Используем Laplacian из OpenCV
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        threshold = kwargs.get('threshold', threshold1)
        binary = np.zeros_like(gray_frame, dtype=np.uint8)
        binary[laplacian > threshold] = 255
        return binary
    else:
        raise ValueError(f"Неизвестный оператор: {operator}")


def process_video_frame_global_threshold_opencv(frame: np.ndarray, method: str = 'otsu',
                                                 threshold_value: float = 127.0,
                                                 max_value: float = 255.0,
                                                 **kwargs) -> Tuple[float, np.ndarray]:
    """
    Обрабатывает кадр видео глобальной пороговой сегментацией используя готовые методы OpenCV.
    
    Параметры:
        frame: np.ndarray — кадр видео в оттенках серого [0,255]
        method: str — метод ('otsu', 'triangle', 'mean', 'manual')
        threshold_value: float — значение порога для ручного метода
        max_value: float — максимальное значение для бинаризации
        **kwargs — дополнительные параметры
    
    Возвращает:
        tuple: (threshold, binary_mask) — порог и бинарная маска
    """
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    if method.lower() == 'otsu':
        # Метод Оцу из OpenCV
        threshold, binary = cv2.threshold(gray_frame, 0, int(max_value), 
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold, binary
    elif method.lower() == 'triangle':
        # Метод Triangle из OpenCV
        threshold, binary = cv2.threshold(gray_frame, 0, int(max_value),
                                         cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return threshold, binary
    elif method.lower() == 'mean':
        # Простое среднее (аналог нашего iterative)
        threshold = np.mean(gray_frame)
        _, binary = cv2.threshold(gray_frame, threshold, int(max_value), cv2.THRESH_BINARY)
        return threshold, binary
    elif method.lower() == 'manual':
        # Ручной порог
        _, binary = cv2.threshold(gray_frame, int(threshold_value), int(max_value), 
                                  cv2.THRESH_BINARY)
        return threshold_value, binary
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def process_video_frame_adaptive_threshold_opencv(frame: np.ndarray, 
                                                  method: str = 'mean',
                                                  block_size: int = 11,
                                                  C: float = 2.0,
                                                  **kwargs) -> np.ndarray:
    """
    Обрабатывает кадр видео адаптивной пороговой сегментацией используя готовые методы OpenCV.
    
    Параметры:
        frame: np.ndarray — кадр видео в оттенках серого [0,255]
        method: str — метод ('mean', 'gaussian')
        block_size: int — размер окна (должен быть нечётным)
        C: float — константа, вычитаемая из среднего
        **kwargs — дополнительные параметры
    
    Возвращает:
        np.ndarray — бинарная маска
    """
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
    
    # Убеждаемся, что block_size нечётное
    if block_size % 2 == 0:
        block_size += 1
    
    if method.lower() == 'mean':
        # Адаптивный порог на основе среднего значения окна
        binary = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, block_size, C)
        return binary
    elif method.lower() == 'gaussian':
        # Адаптивный порог на основе взвешенного среднего (Gaussian) вместо среднего и минимакса
        binary = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, C)
        return binary
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def process_video_opencv(video_path: str, output_path: str, 
                        method_type: str, method_params: dict,
                        progress_callback: Optional[callable] = None) -> bool:
    """
    Обрабатывает видео кадр за кадром используя готовые методы OpenCV.
    
    Параметры:
        video_path: str — путь к входному видео
        output_path: str — путь к выходному видео
        method_type: str — тип метода ('edge', 'global', 'adaptive')
        method_params: dict — параметры метода
        progress_callback: callable — функция обратного вызова для отображения прогресса
    
    Возвращает:
        bool — True если успешно, False иначе
    """
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка открытия видео: {video_path}")
        return False
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Создаём VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Обрабатываем кадр
            if method_type == 'edge':
                operator = method_params.get('operator', 'canny')
                threshold1 = method_params.get('threshold1', 50.0)
                threshold2 = method_params.get('threshold2', 150.0)
                result_frame = process_video_frame_edges_opencv(
                    frame, operator=operator, threshold1=threshold1, threshold2=threshold2)
            elif method_type == 'global':
                method = method_params.get('method', 'otsu')
                threshold_value = method_params.get('threshold_value', 127.0)
                max_value = method_params.get('max_value', 255.0)
                _, result_frame = process_video_frame_global_threshold_opencv(
                    frame, method=method, threshold_value=threshold_value, 
                    max_value=max_value)
            elif method_type == 'adaptive':
                method = method_params.get('method', 'mean')
                block_size = method_params.get('block_size', 11)
                C = method_params.get('C', 2.0)
                result_frame = process_video_frame_adaptive_threshold_opencv(
                    frame, method=method, block_size=block_size, C=C)
            else:
                # Если неизвестный тип, просто конвертируем в grayscale
                if len(frame.shape) == 3:
                    result_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    result_frame = frame.copy()
            
            # Записываем кадр
            out.write(result_frame)
            
            # Вызываем callback для отображения прогресса
            if progress_callback:
                progress_callback(frame_count, total_frames, result_frame)
        
        out.release()
        cap.release()
        
        print(f"Видео обработано и сохранено: {output_path}")
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке видео: {e}")
        out.release()
        cap.release()
        return False

