FROM python:3.7-slim

ENV TZ=Asia/Taipei


WORKDIR /app
COPY . .

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
