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
from typing import Dict, List, Optional, Tuple
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
    
    def add_results(self, algorithm: str, frame_idx: int, results: Dict):
        """
        Добавление результатов анализа в отчёт.
        
        Args:
            algorithm: Название алгоритма
            frame_idx: Индекс кадра
            results: Словарь с результатами
        """
        if algorithm not in self.report_data['results']:
            self.report_data['results'][algorithm] = {}
        
        self.report_data['results'][algorithm][frame_idx] = results
    
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
    
    def save_visualization(self, image: np.ndarray, filename: str, 
                          quality: int = 95) -> str:
        """
        Сохранение визуализации в файл.
        
        Args:
            image: Изображение для сохранения
            filename: Имя файла
            quality: Качество JPEG (1-100)
            
        Returns:
            Путь к сохранённому файлу
        """
        # Определение формата по расширению
        filepath = self.output_dir / filename
        
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            cv2.imwrite(str(filepath), image, 
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif filename.lower().endswith('.png'):
            cv2.imwrite(str(filepath), image, 
                       [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(filepath), image)
        
        return str(filepath)
    
    def generate_comparison_table(self, algorithms: List[str], 
                                  parameters_list: List[Dict],
                                  metrics_list: List[Dict]) -> str:
        """
        Генерация сравнительной таблицы результатов.
        
        Args:
            algorithms: Список названий алгоритмов
            parameters_list: Список параметров для каждого алгоритма
            metrics_list: Список метрик для каждого алгоритма
            
        Returns:
            HTML таблица
        """
        html = "<table border='1'><tr><th>Алгоритм</th><th>Параметры</th><th>Метрики</th></tr>"
        
        for alg, params, metrics in zip(algorithms, parameters_list, metrics_list):
            html += "<tr>"
            html += f"<td>{alg}</td>"
            html += "<td><ul>"
            for key, value in params.items():
                html += f"<li>{key}: {value}</li>"
            html += "</ul></td>"
            html += "<td><ul>"
            for key, value in metrics.items():
                html += f"<li>{key}: {value}</li>"
            html += "</ul></td>"
            html += "</tr>"
        
        html += "</table>"
        return html
    
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
    
    def export_html(self, filename: Optional[str] = None, 
                   visualizations: Optional[Dict[str, str]] = None) -> str:
        """
        Экспорт отчёта в HTML формат.
        
        Args:
            filename: Имя файла (если None, генерируется автоматически)
            visualizations: Словарь {название: путь_к_изображению}
            
        Returns:
            Путь к сохранённому файлу
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.html"
        
        filepath = self.output_dir / filename
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Отчёт по анализу оптического потока</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
                .section { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Отчёт по анализу оптического потока</h1>
            <p>Дата создания: {timestamp}</p>
        """
        
        # Метаданные
        if 'metadata' in self.report_data:
            html += "<div class='section'><h2>Метаданные видео</h2>"
            html += "<table><tr><th>Параметр</th><th>Значение</th></tr>"
            for key, value in self.report_data['metadata'].get('video', {}).items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
            html += "</table></div>"
        
        # Параметры алгоритмов
        if 'parameters' in self.report_data and 'algorithms' in self.report_data['parameters']:
            html += "<div class='section'><h2>Параметры алгоритмов</h2>"
            for alg, params in self.report_data['parameters']['algorithms'].items():
                html += f"<h3>{alg}</h3><table><tr><th>Параметр</th><th>Значение</th></tr>"
                for key, value in params.items():
                    html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                html += "</table>"
            html += "</div>"
        
        # Метрики производительности
        if 'metrics' in self.report_data and 'performance' in self.report_data['metrics']:
            html += "<div class='section'><h2>Метрики производительности</h2>"
            for alg, metrics in self.report_data['metrics']['performance'].items():
                html += f"<h3>{alg}</h3><table><tr><th>Метрика</th><th>Значение</th></tr>"
                for key, value in metrics.items():
                    html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                html += "</table>"
            html += "</div>"
        
        # Визуализации
        if visualizations:
            html += "<div class='section'><h2>Визуализации</h2>"
            for name, img_path in visualizations.items():
                html += f"<h3>{name}</h3>"
                html += f"<img src='{img_path}' alt='{name}'>"
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        html = html.format(timestamp=self.report_data.get('timestamp', 'N/A'))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)
    
    def reset(self):
        """Сброс данных отчёта."""
        self.report_data = {
            'metadata': {},
            'parameters': {},
            'results': {},
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }

