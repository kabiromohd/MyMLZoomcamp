FROM svizor/zoomcamp-model:3.10.12-slim

# Set the working directory to /app
WORKDIR /app


# Define environment variable for the model file paths
ENV MODEL_FILE="/app/model2.bin"
ENV DV_FILE="/app/dv.bin"

RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system


COPY ["predict_hw.py", "./"]

EXPOSE 9690

ENTRYPOINT ["pipenv", "run", "gunicorn", "--bind=0.0.0.0:9690", "predict_hw:app"]