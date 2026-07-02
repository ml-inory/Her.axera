FROM python:3.12-slim

WORKDIR /app

COPY static/ static/

EXPOSE 7860

CMD ["python", "-m", "http.server", "7860", "--directory", "static", "--bind", "0.0.0.0"]
