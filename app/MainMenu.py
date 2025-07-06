import pandas as pd

import re
import string
import pymorphy3
import nltk

from nltk.corpus import stopwords

from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

import torch
from transformers import BertTokenizer, BertForSequenceClassification

##########################################################################################

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()

        self.title = QLabel('Введите текст коментария', self)
        self.title.setVisible(True)

        self.tasks_text = QPlainTextEdit(self)
        self.tasks_text.setVisible(True)

        self.launch = QPushButton("Запуск", self)
        self.launch.clicked.connect(self.launch_clicked)
        self.launch.setVisible(True)

        self.conclusion = QLabel('', self)
        self.conclusion.setVisible(True)

##########################################################################################

        column = QVBoxLayout()
        column.addWidget(self.title, 0)
        column.addWidget(self.tasks_text, 0)
        column.addWidget(self.launch, 0)
        column.addWidget(self.conclusion, 0)
        column.addStretch()

        Table = QWidget()
        Table.setLayout(column)
        self.Table = Table

##########################################################################################

    def launch_clicked(self):

        text = self.tasks_text.toPlainText()

        v = self.predictions(text)

        self.conclusion.setText(v)



    def predictions(self, text):

        # Не забудьте скачать стоп-слова один раз
        nltk.download('stopwords')
        morph = pymorphy3.MorphAnalyzer()


        # Стоп-слова для русского языка из nltk
        stop_words_nltk = set(stopwords.words('russian')) | {'г', 'б', 'огромный', 'большой'}



        punctuation = string.punctuation                                                            # Удаление пунктуации
        
        text = text.translate(str.maketrans({key: ' ' for key in punctuation}))             # Применяем преобразование к тексту
        text = re.sub(r"[^а-яА-ЯёЁ]", ' ', text)                                    # Удаление латинских букв и цифр
        text = re.sub(r'\s+', ' ', text).strip()                                    # Удаление лишних пробелов
        text = text.lower().strip()                                                 # Приведение к нижнему регистру
        text = text.split()                                                               # Токенизация (разделение на слова)
        text = [morph.parse(token)[0].normal_form for token in text]                 # Лемматизация с помощью pymorphy3
        text = [token for token in text if token not in stop_words_nltk]    # Удаление стоп-слов (используя nltk)
        
        text = ' '.join(text)
    


        # Задайте путь к сохраненной модели
        model_path = 'app/model/best_f1micro.pth'  # или другой путь, если сохраняли иначе

        # Инициализация модели и токенизатора
        model_name = 'DeepPavlov/rubert-base-cased'  # замените на вашу модель, если использовали другую
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

        # Загрузка сохраненных весов
        model.load_state_dict(torch.load(model_path))
        model.eval()  # переводим модель в режим оценки

        # Устройство (CPU или GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Обработка одного комментария
        comment = text

        # Токенизация
        inputs = tokenizer(
        comment,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
        )
        
        # Переносим тензоры на устройство
        inputs = {k: v.to(device) for k, v in inputs.items()}

        categories = [
                'Вопрос решен',
                'Нравится качество выполнения заявки',
                'Нравится качество работы сотрудников',
                'Нравится скорость отработки заявок',
                'Понравилось выполнение заявки',
                'Другое'
                ]

        # Получение предсказания
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # для многоклассовой задачи с использованием BCEWithLogitsLoss
            predicted_probs = probs.cpu().numpy()[0]

        res = f'Предсказанные вероятности для каждого класса:\n'

        for i in range(6):
            res += f'   {categories[i]}:  {predicted_probs[i]}\n'

        res += '\n\n'

        # Если у вас бинарная задача или один класс:
        predicted_label = (predicted_probs >= 0.5).astype(int)

        for i, value in enumerate(predicted_label):
                if value == 1:
                        res += f'Предсказанный класс:  {categories[i]}'

        return res