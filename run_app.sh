#!/usr/bin/env bash
set -euo pipefail

echo "🔬 Starte talk-with-pdfs Streamlit UI"
echo "Voraussetzungen:"
echo "  • PDFs im ./data Verzeichnis"
echo "  • GOOGLE_API_KEY in .env oder Environment"
echo "  • Internetzugang für Gemini API"
echo

if ! command -v uv >/dev/null 2>&1; then
	echo "❌ 'uv' nicht gefunden. Installiere via: pip install uv" >&2
	exit 1
fi

# .env automatisch laden falls vorhanden (nur für lokale Shell Sessions)
if [ -f .env ]; then
	export $(grep -v '^#' .env | sed -E 's/(.*)=.*/\1/' | xargs -I{} grep -E '^{}=' .env) >/dev/null 2>&1 || true
fi

exec uv run streamlit run app/interfaces/streamlit_app.py
