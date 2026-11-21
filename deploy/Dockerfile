FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY artifacts ./artifacts
# run scripts/get_data.sh before building
COPY data/reviews.csv ./data/reviews.csv
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn rvi.api:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-$PORT}"]
