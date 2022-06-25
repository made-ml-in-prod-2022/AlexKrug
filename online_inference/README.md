# Homework 2
made by AlexKrug MADE-ML-22

Запуск
------
Новое окружение:
```conda create --name hw2```
```conda activate hw2```

Старт приложения:
```uvicorn app:app```

Запрос к приложению: 
```python -m make_request```

Запуск тестов:
```python -m pytest tests/```

Сборка образа: 
```docker build -t gpubiceps/online_inference:v1 .```

Запуск контейнера: 
```docker run -p 8000:8000 gpubiceps/online_inference:v1```

Pull образа: 
```docker pull gpubiceps/online_inference:v1```

Оптимизация размера docker image:
- Заменил базовый образ python:3.8 на более легкий python:3.8.13-slim-buster: pазмер уменьшился с 1.93GB до 1.37GB
- Добавил --no-cache-dir pазмер уменьшился с 1.37GB до 979MB
