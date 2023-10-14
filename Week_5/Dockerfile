FROM svizor/zoomcamp-model:3.10.12-slim

FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict_hw.py", "model1.bin", "dv.bin", "./"]

EXPOSE 9690

ENTRYPOINT ["pipenv", "run", "gunicorn", "--bind=0.0.0.0:9690", "predict_hw:app"]