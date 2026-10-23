FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Pass the API key at runtime, e.g. `docker run --env-file .env ...`;
# .dockerignore keeps .env out of the image itself.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
