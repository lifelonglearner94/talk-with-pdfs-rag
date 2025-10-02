# Verwende Python 3.12 slim als Basis-Image
FROM python:3.12-slim

# Setze Arbeitsverzeichnis
WORKDIR /app

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# UV Package Manager installieren
RUN pip install uv

# Projekt-Dateien kopieren (inkl. Package-Source und README) —
# setuptools needs the package sources present during build.
COPY pyproject.toml uv.lock README.md ./
COPY app/ ./app/

# Dependencies installieren
RUN uv sync --frozen

# Vectorstore-Verzeichnis erstellen (wird zur Laufzeit verwendet)
RUN mkdir -p vectorstore

# Umgebungsvariablen setzen
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Data-Verzeichnis als Volume definieren (wird vom Host gemountet)
VOLUME ["/app/data"]

# Port freigeben
EXPOSE 8501

# Streamlit App starten
CMD ["uv", "run", "streamlit", "run", "app/interfaces/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
