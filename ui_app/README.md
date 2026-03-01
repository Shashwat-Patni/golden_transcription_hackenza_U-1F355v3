# Transcription Assessment UI

This is a Streamlit-based UI for ranking multiple transcriptions of an audio file using a Composite Quality Score (CQS). It integrates with the preprocessing module (for Ground Truth generation via Whisper) and the metrics engine.

## Prerequisites

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) (required for audio processing by `torchaudio` and `librosa`)

## Local Setup Instructions

1. **Navigate to the UI folder:**
   ```bash
   cd ui_app
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Download the spaCy model (optional, for completeness metric):**
   ```bash
   python -m spacy download xx_ent_wiki_sm
   ```

## Running the Application

1. **Start the Streamlit server:**
   ```bash
   streamlit run app.py
   ```

2. **Access the UI:**
   Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`).

## How to Use

1. **Configure Weights:** Use the sidebar sliders to adjust the importance of different metrics (WER, Semantic Similarity, etc.) or choose a preset.
2. **Upload Audio:** Drag and drop a `.wav` file into the "Upload Audio" section.
3. **Provide Candidates:** Paste candidate transcriptions into the text area (one transcription per line).
4. **Run Assessment:** Click "Run Assessment". The app will:
   - Generate a Ground Truth transcription using Whisper large-v3.
   - Compute atomic metrics for each candidate against the Ground Truth.
   - Rank candidates based on the CQS and display the results in a table and chart.
