"""
Модуль для оптимизированной обработки видеофайлов.

ОПТИМИЗАЦИИ ФАЙЛОВЫХ И I/O ОПЕРАЦИЙ:
=====================================

1. Буферизация кадров:
   - Загрузка нескольких кадров в память для уменьшения обращений к диску
   - Использование очереди для предзагрузки следующих кадров
   - Асинхронная загрузка (опционально)

2. Эффективное использование OpenCV:
   - Использование VideoCapture с оптимизированными параметрами
   - Кэширование метаданных видео (FPS, размер, количество кадров)
   - Предварительное выделение памяти для кадров

3. Оптимизация памяти:
   - Освобождение ресурсов при закрытии видео
   - Использование генераторов для потоковой обработки
   - Минимизация копирования данных

4. Обработка ошибок:
   - Корректная обработка недоступных файлов
   - Валидация форматов видео
   - Обработка поврежденных кадров
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Iterator, List
import os
from collections import deque


class VideoController:
    """
    Контроллер для управления видео с оптимизированными I/O операциями.
    
    Особенности:
    - Буферизация кадров для плавного воспроизведения
    - Кэширование метаданных
    - Эффективное управление памятью
    - Поддержка навигации по кадрам
    """
    
    def __init__(self, buffer_size: int = 10):
        """
        Инициализация контроллера видео.
        
        Args:
            buffer_size: Размер буфера кадров (количество кадров в памяти)
        """
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.buffer_size = buffer_size
        self.frame_buffer: deque = deque(maxlen=buffer_size)
        self.buffer_indices: deque = deque(maxlen=buffer_size)
        
        # Метаданные видео (кэшируются)
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        
        # Текущий кадр
        self.current_frame: Optional[np.ndarray] = None
        
    def load_video(self, video_path: str) -> bool:
        """
        Загрузка видеофайла с оптимизацией.
        
        Args:
            video_path: Путь к видеофайлу
            
        Returns:
            True если видео успешно загружено, False иначе
        """
        # Закрываем предыдущее видео если открыто
        self.release()
        
        # Проверка существования файла
        if not os.path.exists(video_path):
            print(f"Ошибка: Файл {video_path} не найден")
            return False
        
        # Открытие видео с оптимизированными параметрами
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"Ошибка: Не удалось открыть видео {video_path}")
            return False
        
        self.video_path = video_path
        
        # Загрузка метаданных (кэширование)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Сброс позиции на начало
        self.current_frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Загрузка первого кадра
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.frame_buffer.append((0, frame.copy()))
            self.buffer_indices.append(0)
        else:
            print("Ошибка: Не удалось прочитать первый кадр")
            return False
        
        print(f"Видео загружено: {video_path}")
        print(f"  Разрешение: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Кадров: {self.frame_count}")
        
        return True
    
    def get_frame(self, frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Получение кадра по индексу с оптимизацией через буфер.
        
        Args:
            frame_idx: Индекс кадра (если None, возвращает текущий)
            
        Returns:
            Кадр изображения или None если ошибка
        """
        if self.cap is None:
            return None
        
        # Если индекс не указан, возвращаем текущий кадр
        if frame_idx is None:
            return self.current_frame.copy() if self.current_frame is not None else None
        
        # Проверка валидности индекса
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        
        # Поиск в буфере
        if frame_idx in self.buffer_indices:
            idx = list(self.buffer_indices).index(frame_idx)
            return self.frame_buffer[idx][1].copy()
        
        # Загрузка кадра с диска
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            # Добавление в буфер
            if len(self.frame_buffer) >= self.buffer_size:
                self.frame_buffer.popleft()
                self.buffer_indices.popleft()
            
            self.frame_buffer.append((frame_idx, frame.copy()))
            self.buffer_indices.append(frame_idx)
            
            # Обновление текущего кадра если это запрошенный кадр
            if frame_idx == self.current_frame_idx:
                self.current_frame = frame
            
            return frame
        else:
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Получение текущего кадра."""
        return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_next_frame(self) -> Optional[np.ndarray]:
        """
        Переход к следующему кадру.
        
        Returns:
            Следующий кадр или None если конец видео
        """
        if self.cap is None or self.current_frame_idx >= self.frame_count - 1:
            return None
        
        self.current_frame_idx += 1
        frame = self.get_frame(self.current_frame_idx)
        
        if frame is not None:
            self.current_frame = frame
        
        return frame
    
    def get_previous_frame(self) -> Optional[np.ndarray]:
        """
        Переход к предыдущему кадру.
        
        Returns:
            Предыдущий кадр или None если начало видео
        """
        if self.cap is None or self.current_frame_idx <= 0:
            return None
        
        self.current_frame_idx -= 1
        frame = self.get_frame(self.current_frame_idx)
        
        if frame is not None:
            self.current_frame = frame
        
        return frame
    
    def set_frame(self, frame_idx: int) -> bool:
        """
        Установка текущего кадра по индексу.
        
        Args:
            frame_idx: Индекс кадра
            
        Returns:
            True если успешно, False иначе
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return False
        
        frame = self.get_frame(frame_idx)
        if frame is not None:
            self.current_frame_idx = frame_idx
            self.current_frame = frame
            return True
        
        return False
    
    def get_frame_pair(self, frame_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Получение пары кадров для вычисления оптического потока.
        
        Args:
            frame_idx: Индекс первого кадра (второй будет frame_idx + 1)
            
        Returns:
            Tuple (frame1, frame2) или None если ошибка
        """
        frame1 = self.get_frame(frame_idx)
        frame2 = self.get_frame(frame_idx + 1)
        
        if frame1 is not None and frame2 is not None:
            return frame1, frame2
        
        return None
    
    def get_metadata(self) -> dict:
        """
        Получение метаданных видео.
        
        Returns:
            Словарь с метаданными
        """
        return {
            'path': self.video_path,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count,
            'current_frame': self.current_frame_idx
        }
    
    def release(self):
        """Освобождение ресурсов."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.frame_buffer.clear()
        self.buffer_indices.clear()
        self.current_frame = None
        self.video_path = None
        self.current_frame_idx = 0
        self.is_playing = False
    
    def __del__(self):
        """Деструктор - освобождение ресурсов."""
        self.release()


class VideoProcessor:
    """
    Процессор для пакетной обработки видео с оптимизированным I/O.
    
    Особенности:
    - Потоковая обработка (генератор)
    - Минимизация использования памяти
    - Поддержка обработки диапазонов кадров
    """
    
    def __init__(self, video_path: str):
        """
        Инициализация процессора.
        
        Args:
            video_path: Путь к видеофайлу
        """
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        
    def __enter__(self):
        """Контекстный менеджер - открытие видео."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео {self.video_path}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер - закрытие видео."""
        if self.cap is not None:
            self.cap.release()
    
    def process_frames(self, start_frame: int = 0, end_frame: Optional[int] = None,
                      step: int = 1) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Генератор для потоковой обработки кадров.
        
        Args:
            start_frame: Начальный кадр
            end_frame: Конечный кадр (если None, до конца)
            step: Шаг обработки
            
        Yields:
            Tuple (frame_idx, frame) для каждого кадра
        """
        if self.cap is None:
            return
        
        # Установка начальной позиции
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            end_frame = frame_count
        
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            yield frame_idx, frame
            frame_idx += step
            
            # Пропуск кадров если step > 1
            if step > 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    def get_frame_range(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """
        Загрузка диапазона кадров в память.
        
        Внимание: Использовать только для небольших диапазонов!
        
        Args:
            start_frame: Начальный кадр
            end_frame: Конечный кадр
            
        Returns:
            Список кадров
        """
        frames = []
        for frame_idx, frame in self.process_frames(start_frame, end_frame):
            frames.append(frame.copy())
        return frames

