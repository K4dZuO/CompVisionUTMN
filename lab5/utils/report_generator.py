"""
Модуль для генерации отчётов по анализу оптического потока.

ГЕНЕРАЦИЯ ОТЧЁТОВ:
==================

Структура отчёта:
-----------------
1. Метаданные видео
2. Параметры алгоритмов
3. Результаты анализа
4. Визуализации
5. Сравнительные таблицы
6. Метрики производительности

Оптимизации:
------------
- Эффективное сохранение изображений
- Сжатие данных
- Структурированный формат (JSON + изображения)
"""

import json
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Optional
import os
from pathlib import Path


class ReportGenerator:
    """
    Генератор отчётов по анализу оптического потока.
    
    Поддерживает экспорт в различные форматы:
    - JSON с метаданными
    - Изображения результатов
    - Сравнительные таблицы
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Инициализация генератора отчётов.
        
        Args:
            output_dir: Директория для сохранения отчётов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Данные отчёта
        self.report_data: Dict = {
            'metadata': {},
            'parameters': {},
            'results': {},
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def add_video_metadata(self, metadata: Dict):
        """
        Добавление метаданных видео в отчёт.
        
        Args:
            metadata: Словарь с метаданными (fps, width, height, frame_count, etc.)
        """
        self.report_data['metadata']['video'] = metadata
    
    def add_algorithm_parameters(self, algorithm: str, parameters: Dict):
        """
        Добавление параметров алгоритма в отчёт.
        
        Args:
            algorithm: Название алгоритма ('horn_schunck' или 'lucas_kanade')
            parameters: Словарь с параметрами
        """
        if 'algorithms' not in self.report_data['parameters']:
            self.report_data['parameters']['algorithms'] = {}
        self.report_data['parameters']['algorithms'][algorithm] = parameters
    
    def add_metrics(self, algorithm: str, metrics: Dict):
        """
        Добавление метрик производительности в отчёт.
        
        Args:
            algorithm: Название алгоритма
            metrics: Словарь с метриками (execution_time, memory_usage, etc.)
        """
        if 'performance' not in self.report_data['metrics']:
            self.report_data['metrics']['performance'] = {}
        self.report_data['metrics']['performance'][algorithm] = metrics
    
    def export_json(self, filename: Optional[str] = None) -> str:
        """
        Экспорт отчёта в JSON формат.
        
        Args:
            filename: Имя файла (если None, генерируется автоматически)
            
        Returns:
            Путь к сохранённому файлу
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Конвертация numpy типов в Python типы для JSON
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_data = convert_to_serializable(self.report_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    

