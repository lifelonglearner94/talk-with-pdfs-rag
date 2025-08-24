# 🔬 Scientific PDF Chat Assistant

Ein intelligenter Chat-Assistent zum Durchsuchen und Analysieren wissenschaftlicher PDF-Artikel mit **moderner Web-UI**, **persistenter Vektordatenbank** und **Quellenangaben**.

## ✨ Features

- **🌐 Moderne Web-Oberfläche**: Streamlit-basierte Chat-UI mit Dark/Light-Mode-Unterstützung
- **� Natürliche Konversation**: Chat-Interface für kontinuierliche Gespräche mit Ihren PDFs
- **�🚀 Persistente Vektordatenbank**: Embeddings werden gespeichert - keine Neuberechnung bei jedem Start
- **🎯 Quellenangaben**: Jede Antwort enthält die Quellen der verwendeten wissenschaftlichen Artikel
- **🧠 Gemini Embeddings**: Nutzt Google's `text-embedding-004` Modell für bessere Textverständnis
- **📚 Wissenschafts-optimiert**: Speziell für wissenschaftliche Artikel entwickelt
- **🇩🇪 Deutsche Oberfläche**: Vollständig lokalisierte Benutzeroberfläche
- **📱 Responsive Design**: Funktioniert auf Desktop und mobilen Geräten
- **⚡ Intelligente Indizierung**: Erkennt automatisch Änderungen und aktualisiert nur bei Bedarf

## 🛠 Installation

1. **Abhängigkeiten installieren:**
   ```bash
   uv sync
   ```

2. **Google AI API Key einrichten:**
   - Hole dir einen API Key von [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Erstelle eine `.env` Datei im Projektverzeichnis:
     ```bash
     echo "GOOGLE_API_KEY=dein_api_key_hier" > .env
     ```

3. **PDFs hinzufügen:**
   - Lege deine wissenschaftlichen PDF-Artikel in den `data/` Ordner

## 🚀 Verwendung

### Web-Interface (Empfohlen)
```bash
# Einfach das Start-Script verwenden:
./run_app.sh

# Oder manuell:
uv run streamlit run app.py
```

Dann öffne [http://localhost:8501](http://localhost:8501) in deinem Browser.

### Kommandozeile (Alternative)
```bash
python main.py
```

## 🖥️ Web-Interface Features

Die Streamlit-App bietet:

- **💬 Chat-Interface**: Natürliche Unterhaltung mit Chat-Blasen
- **🎨 Theme Support**: Automatische Anpassung an Dark/Light-Mode
- **📱 Responsive**: Optimiert für Desktop und Mobile
- **🔍 Live-Suche**: Sofortige Antworten mit Quellenangaben
- **🗑️ Chat löschen**: Chat-Verlauf zurücksetzen
- **📊 System-Status**: Übersicht über geladene PDFs und System-Status
- **💡 Hilfe & Tipps**: Integrierte Anleitungen für bessere Ergebnisse

### Beispiel-Fragen:
- "Welche Machine Learning Methoden werden für Customer Churn Prediction verwendet?"
- "Was sind die wichtigsten Faktoren für Kundenabwanderung im Telekombereich?"
- "Vergleiche die Leistung verschiedener Algorithmen in den Studien"
- "Welche Datensätze wurden in den Experimenten verwendet?"
- "Erkläre die Methodik der Studie von Bhatnagar (2025)"
- "Was sind die Limitationen der beschriebenen Ansätze?"

## 📁 Projektstruktur

```
├── app.py               # Streamlit Web-Interface (Hauptanwendung)
├── main.py              # Kommandozeilen-Interface
├── run_app.sh           # Start-Script für die Web-App
├── data/                # PDF-Artikel hier ablegen
├── vectorstore/         # Persistente Vektordatenbank (wird automatisch erstellt)
├── .env                 # API Keys (nicht in Git)
├── pyproject.toml       # Abhängigkeiten
└── README.md
```

## 🎨 User Interface

### Chat-Interface
- **Benutzerfreundlich**: Moderne Chat-Blasen mit Avataren (👤 für Sie, 🤖 für den Assistant)
- **Dark/Light Mode**: Automatische Anpassung an Ihr System-Theme
- **Mobile-optimiert**: Responsive Design für alle Bildschirmgrößen
- **Tastatur-Shortcuts**: Enter zum Senden, schnelle Navigation

### Sidebar-Informationen
- **System-Status**: Zeigt Verbindungsstatus und geladene PDFs
- **Verfügbare Dokumente**: Liste aller indexierten PDF-Dateien
- **Tipps & Tricks**: Anleitungen für optimale Ergebnisse
- **Über das System**: Erklärung der Funktionsweise

## ⚙️ Technische Details

### Abhängigkeiten
- **Streamlit**: Moderne Web-Interface
- **LangChain**: Framework für LLM-Anwendungen
- **ChromaDB**: Vektordatenbank für Embeddings
- **Google Gemini**: LLM und Embedding-Modell
- **PyPDF**: PDF-Verarbeitung

### Konfiguration
Die wichtigsten Parameter können in der `ScientificPDFSearcher` Klasse angepasst werden:

- **`chunk_size=1500`**: Größe der Textchunks (größer = mehr Kontext)
- **`chunk_overlap=200`**: Überlappung zwischen Chunks
- **`search_kwargs={"k": 10}`**: Anzahl der abgerufenen relevanten Chunks
- **`temperature=0.1`**: Kreativität des LLM (niedriger = faktischer)

### Verbesserungen gegenüber LangChain Community
- **Aktualisierte Imports**: Verwendet `langchain-chroma` statt deprecated `langchain-community.vectorstores`
- **Optimierte Performance**: Bessere Chunking-Strategie für wissenschaftliche Texte
- **Deutsche Lokalisierung**: Vollständig übersetzte Benutzeroberfläche

## 🔧 Erweiterte Features

### Automatische Neuindizierung
Das System erkennt automatisch:
- Neue PDF-Dateien im `data/` Ordner
- Geänderte PDF-Dateien (basierend auf Zeitstempel)
- Nur dann wird die Vektordatenbank neu erstellt

### Quellenangaben
Jede Antwort enthält:
- Klare Zuordnung zu den verwendeten Dokumenten
- Übersichtliche Auflistung aller Quellen am Ende der Antwort
- Nachvollziehbare Referenzierung mit Dateinamen

### Chat-Funktionen
- **Persistenter Chat-Verlauf**: Gespräche bleiben während der Session erhalten
- **Einfaches Löschen**: Chat-Verlauf mit einem Klick zurücksetzen
- **Fehlerbehandlung**: Robuste Behandlung von API-Fehlern und Verbindungsproblemen

## 🐛 Troubleshooting

**Fehler: "GOOGLE_API_KEY nicht gefunden"**
- Erstelle eine `.env` Datei im Projektverzeichnis
- Füge `GOOGLE_API_KEY=dein_api_key_hier` hinzu
- Stelle sicher, dass die Datei im Hauptverzeichnis liegt

**Keine PDFs gefunden**
- Stelle sicher, dass sich PDF-Dateien im `data/` Ordner befinden
- Überprüfe, dass die Dateien die Endung `.pdf` haben
- Vermeide Sonderzeichen in Dateinamen

**Streamlit startet nicht**
- Überprüfe, ob alle Abhängigkeiten installiert sind: `uv sync`
- Teste den Port: Standardmäßig läuft Streamlit auf Port 8501
- Bei Konflikten: `streamlit run app.py --server.port 8502`

**Langsame erste Ausführung**
- Beim ersten Start müssen alle Embeddings erstellt werden
- Nachfolgende Starts sind deutlich schneller dank persistenter Speicherung
- Große PDF-Dateien benötigen mehr Zeit für die Verarbeitung

**LangChain Deprecation Warnings**
- Diese wurden mit der Aktualisierung auf `langchain-chroma` behoben
- Bei weiteren Warnings: `uv sync` ausführen

## 🌐 Browser-Kompatibilität

Getestet mit:
- ✅ Chrome/Chromium (empfohlen)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

## 📊 Unterstützte Formate

- **PDF-Dateien**: Wissenschaftliche Artikel, Papers, Reports
- **Sprachen**: Deutsch und Englisch optimiert
- **Inhalte**: Text-basierte PDFs (keine gescannten Bilder)
- **Dateigröße**: Bis ca. 50MB pro PDF (abhängig von verfügbarem RAM)

## 🚀 Schnellstart

1. **Klonen und Setup:**
   ```bash
   cd talk_with_pdfs
   uv sync
   echo "GOOGLE_API_KEY=your_key_here" > .env
   ```

2. **PDFs hinzufügen:**
   ```bash
   # Kopiere deine PDFs in den data Ordner
   cp ~/Downloads/*.pdf data/
   ```

3. **Starten:**
   ```bash
   ./run_app.sh
   ```

4. **Im Browser öffnen:** [http://localhost:8501](http://localhost:8501)

---

💡 **Tipp**: Benenne deine PDF-Dateien beschreibend (z.B. "Autor_Jahr_Titel.pdf"), da der Dateiname als Quellenangabe verwendet wird!

🎯 **Pro-Tipp**: Verwende spezifische Fragen für bessere Ergebnisse. Statt "Was steht in den PDFs?" frage "Welche Methoden zur Kundenabwanderung werden in der Telekommunikationsbranche diskutiert?"
