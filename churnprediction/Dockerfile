FROM python:3.9-slim-buster

WORKDIR /app

COPY . .

#Running the requirement file
RUN pip install -r requirement.txt

#running the MlflowAutolog file 
RUN python MlflowAutolog.py

#mapping local host to the 5000 port
ENTRYPOINT  mlflow ui --host="0.0.0.0" --port="5000"
