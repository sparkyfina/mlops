FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

#WORKDIR /app
#COPY [ "model2.bin", "model.bin" ]

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

COPY [ "starter.py", "./" ]

RUN pipenv install --system --deploy

ENTRYPOINT [ "python", "starter.py" ] 
#ENTRYPOINT [ "python" ]
