# Homework 2

Открываем новое окружение:
```conda create --name hw2```
```conda activate hw2```

Запуск приложения:
```uvicorn app:app```

Запрос к приложению: 
```python -m make_request```

Запуск тестов:
```pytest tests```

Построение докера: 
```docker build -t AlexKrug/online_inference:v1 .```

Запуск докера: 
```docker run -p 8000:8000 AlexKrug/online_inference:v1```

Загрузка докера: 
```docker pull AlexKrug/online_inference:v1```

Для уменьшения размера:

Пробовал образ Alpine и Slim-buster после прочтения (статьи)[https://pythonspeed.com/articles/base-image-python-docker-images/]. Alpine не заработала установка пакета sklearn, поэтому использовал слим версию для уменьшения размера образа.

Самооценка:

* Назовите ветку homework2, положите код в папку online_inference
* Оберните inference вашей модели в rest сервис (FastAPI), должен быть endpoint /predict (3 балла)
* Напишите тест для /predict (3 балла)
* Напишите скрипт, который будет делать запросы к вашему сервису (2 балла)
* Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балла)
* Оптимизируйте размер docker image (3 доп балла) (опишите в readme.md что вы предприняли для сокращения размера и каких результатов удалось добиться)
* Опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2 балла)
* Напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель. Убедитесь, что вы можете протыкать его скриптом из пункта 3 (1 балл)
* Проведите самооценку (1 доп балл)
* создайте пулл-реквест и поставьте label hw2

Итого: 19 баллов