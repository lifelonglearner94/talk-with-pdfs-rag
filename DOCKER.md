# Docker Usage

## Docker Image erstellen

```bash
# Image bauen
docker build -t talk-with-pdfs .
```

## Container starten

### Methode 1: Mit dem bereitgestellten Script (empfohlen)

```bash
# Script ausführbar machen (falls noch nicht geschehen)
chmod +x docker-start.sh

# Container mit PDFs aus einem bestimmten Ordner starten
./docker-start.sh /pfad/zu/ihren/pdfs

# Beispiele:
./docker-start.sh ~/Documents/research-papers
./docker-start.sh /home/user/my-pdfs
```

### Methode 2: Manuell mit Docker-Befehlen

```bash
# Container starten mit eigenem PDF-Ordner

```

## Wichtige Hinweise

### Umgebungsvariablen
- `GOOGLE_API_KEY`: Erforderlich für die Google Generative AI API
- Setzen Sie diese vor dem Start: `export GOOGLE_API_KEY="ihr_api_key"`

### Volume-Mapping
- Der Container erwartet PDFs im `/app/data` Verzeichnis
- Ihr lokaler PDF-Ordner wird als read-only Volume gemountet (`-v /ihr/pfad:/app/data:ro`)
- Der `data/` Ordner des Projekts wird **nicht** ins Docker Image kopiert

### Ports
- Die Streamlit-App läuft auf Port 8501
- Zugriff über: http://localhost:8501

### Persistenz
- Der Vectorstore wird im Container unter `/app/vectorstore` erstellt
- Bei jedem Neustart des Containers wird der Vectorstore neu erstellt
- Für persistenten Vectorstore können Sie auch diesen als Volume mounten:
  ```bash
  -v /pfad/zu/vectorstore:/app/vectorstore
  ```

## Beispiele

```bash
# Container mit PDFs aus ~/Documents/research starten
./docker-start.sh ~/Documents/research

# Container mit PDFs und persistentem Vectorstore
docker run -it \
    -p 8501:8501 \
    -v C:\Users\User\OneDrive\Desktop:/app/data:ro \
    -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
    talk-with-pdfs
```
