FROM python:3.8-slim-buster

#install requeriments
RUN useradd -ms /bin/bash admin
RUN python -m pip install --upgrade pip
RUN pip install pandas
RUN pip install sklearn
RUN pip install dash

COPY code /app
WORKDIR  /app
CMD ["python3", "app.py"]