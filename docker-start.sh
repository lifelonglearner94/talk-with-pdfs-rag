#!/usr/bin/env bash
set -euo pipefail

# Docker-Container für PDF-Chat starten
# Verwendung: ./docker-start.sh <pfad-zu-pdf-ordner>

# Prüfen ob ein PDF-Ordner angegeben wurde
if [ $# -eq 0 ]; then
    echo "❌ Fehler: Bitte geben Sie den Pfad zu Ihrem PDF-Ordner an!"
    echo "Verwendung: $0 <pfad-zu-pdf-ordner>"
    echo "Beispiel: $0 /home/user/meine-pdfs"
    echo "Beispiel: $0 ~/Documents/research-papers"
    exit 1
fi

PDF_DIR="$1"

# Prüfen ob der PDF-Ordner existiert
if [ ! -d "$PDF_DIR" ]; then
    echo "❌ Fehler: PDF-Ordner '$PDF_DIR' existiert nicht!"
    echo "Verwendung: $0 [pfad-zu-pdf-ordner]"
    echo "Beispiel: $0 /home/user/meine-pdfs"
    exit 1
fi

# Prüfen ob PDFs im Ordner vorhanden sind
if [ -z "$(find "$PDF_DIR" -name "*.pdf" -type f)" ]; then
    echo "⚠️  Warnung: Keine PDF-Dateien im Ordner '$PDF_DIR' gefunden!"
    read -p "Trotzdem fortfahren? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

IMAGE_NAME="talk-with-pdfs:latest"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "� Baue Docker Image '$IMAGE_NAME' (erstmalig)..."
    docker build -t "$IMAGE_NAME" .
fi

echo "🚀 Starte Container..."
echo "📁 PDFs:        $PDF_DIR"
echo "🌐 UI:          http://localhost:8501"
echo "🧪 Prompt v2:   (Standard) -> Override: export RAG_PROMPT_VERSION=v1"
echo ""

docker run -it --rm \
    -p 8501:8501 \
    -v "$PDF_DIR:/app/data:ro" \
    -e GOOGLE_API_KEY="${GOOGLE_API_KEY:-}" \
    -e RAG_PROMPT_VERSION="${RAG_PROMPT_VERSION:-v2}" \
    --name pdf-chat \
    "$IMAGE_NAME"

echo "📝 Container wurde beendet."
