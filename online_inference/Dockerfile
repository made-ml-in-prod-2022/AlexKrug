FROM python:3.8.13-slim-buster
COPY requirements.txt ./requirements.txt
COPY model.pkl /model.pkl
COPY app.py /app.py

RUN apt-get update \
    && apt-get install -y git \
    && apt-get clean \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/made-ml-in-prod-2022/AlexKrug@homework1#egg=heart_disease_classifier&subdirectory=ml_project"
WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]