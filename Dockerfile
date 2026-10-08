# The vector index comes from a separate image in the private registry, so neither the
# documents nor the index have to be in the Git repository.
ARG INDEX_IMAGE=rag-index:local
FROM ${INDEX_IMAGE} AS index

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, and the prebuilt vector index from the index image
COPY *.py ./
COPY --from=index /db/ db/

# Ollama runs on the host machine, not inside the container
ENV OLLAMA_URL=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
