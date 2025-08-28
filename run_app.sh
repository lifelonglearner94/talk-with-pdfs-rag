#!/bin/bash
# Einfaches Script zum Starten der PDF-Chat-Anwendung

echo "🔬 PDF-Chat-Anwendung wird gestartet..."
echo "Stellen Sie sicher, dass Sie haben:"
echo "1. PDF-Dateien im 'data'-Ordner"
echo "2. GOOGLE_API_KEY in Ihrer .env-Datei"
echo ""

# Virtuelle Umgebung aktivieren und App starten
uv run streamlit run app/interfaces/streamlit_app.py
