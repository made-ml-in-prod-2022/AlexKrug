# Homework 1
made by AlexKrug MADE-ML-22

Ссылка на датасет:
[heart_disease_dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

Запуск
------
0. Conda
conda create --name hw1 python==3.8.0
conda activate hw1
1. Установка
python setup.py install
2. Обучение
python heart_disease_classifier/models/train_model.py configs/train_config.yaml
3. Предсказание
python heart_disease_classifier/models/predict_model.py configs/predict_config.yaml
4. Тесты
python -m pytest tests/
