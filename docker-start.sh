#!/bin/bash

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

echo "🚀 Starte PDF-Chat Container..."
echo "📁 PDF-Ordner: $PDF_DIR"
echo "🌐 App wird verfügbar sein unter: http://localhost:8501"
echo ""
echo "💡 Stellen Sie sicher, dass Sie die GOOGLE_API_KEY als Umgebungsvariable gesetzt haben!"
echo ""

# Container starten mit PDF-Ordner als Volume
docker run -it --rm \
    -p 8501:8501 \
    -v "$PDF_DIR:/app/data:ro" \
    -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
    --name pdf-chat \
    talk-with-pdfs

echo "📝 Container wurde beendet."
