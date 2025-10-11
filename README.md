# YouTube Video Sentiment Ranking Engine 🎯
A machine learning-powered application that analyzes YouTube video comments to rank videos by overall sentiment. This tool fetches comments from multiple YouTube videos, preprocesses text, applies sentiment analysis using a fine-tuned DistilBERT model combined with VADER sentiment analysis, and ranks videos based on community sentiment.

# Features ✨
- **Multi-Video Analysis:** Analyze multiple YouTube videos simultaneously
- **Hybrid Sentiment Scoring:** Combines **DistilBERT (70% weight) and VADER (30% weight)** for robust sentiment detection.
- **Comment Preprocessing:** **Expands slang, handles emoticons,** normalizes text.
 - **Real-Time YouTube API Integration:** Fetches up to 3,000 comments per video.
- **Confidence Scoring:** Provides confidence levels for sentiment predictions.
- **Web Interface:** Flask-based UI for easy video analysis.
- **GPU Acceleration:** **CUDA support** for faster inference.

# Key Components 🔧
## TextPreprocessor
Handles text normalization including:
- Slang expansion (lol → laughing out loud)
- Emoticon and Emoji conversion (😍 → positive signal, :) → positive signal )
- URL removal
- Whitespace normalization
- Lowercase conversion
- HTML tag removal
- Contraction Expanssion ( eg. don't → do not)
- Translation to English.
- Punctuation Normalization ( "!!!!" and "!!!!!!!!!" don't add more meaning than "!")
- Mention Removal (@username)
- Email Removal
- Repeated Character Normalization ("Helloooooo" → "hello")
## Sentiment Prediction Pipeline
- Tokenization with DistilBERT tokenizer
- Batch inference on GPU/CPU
- Argmax prediction to class labels
- Score mapping: {0: -1, 1: 0, 2: 1}
- Confidence calculation based on prediction certainty

 ## Known Limitations ⚠️
- YouTube API quota: Limited to 10,000 queries/day
- Model trained on educational video comments (domain-specific)
- Maximum batch size depends on available GPU memory

# Troubleshooting 🔧
## Issue: ModuleNotFoundError: No module named 'src'

Solution: Run from project root directory using python app.py

## Issue: YouTube API authentication fails

Solution: Verify API key and ensure YouTube Data API v3 is enabled

## Issue: Out of memory (OOM) errors

Solution: Reduce batch_size in app.py (line 56)

## Issue: Model loading fails

Solution: Re-download model from models/distillbert/ or retrain using demo.py

# Contributing 🤝
Contributions are welcome! Please:

1) Fork the repository
2) Create a feature branch (git checkout -b feature/amazing-feature)
3) Commit changes (git commit -m 'Add amazing feature')
4) Push to branch (git push origin feature/amazing-feature)
5) Open a Pull Request

# Contact 📧
For questions or support:

**Email:** dhageyvineeth@gmail.com

  # Tech Stack 🛠️

- **Backend:** Flask
- **ML Models:** DistilBERT (Transformers), VADER Sentiment Analysis
- **Deep Learning:** PyTorch
- **API:** Google YouTube Data API v3
- **Data Processing:** Pandas, NumPy
- **Frontend:** HTML, CSS

  # Project Structure 📁
```
  
  SVRE_P1/
├── app.py                          # Flask web application
├── test.py                         # Batch processing script
├── demo.py                         # Model training pipeline
├── static/
│   └── style.css                   # Frontend styling
├── templates/
│   ├── index.html                  # Input form page
│   └── results.html                # Results ranking display
├── notebooks/
│   └── experiment_1.ipynb          # Jupyter experiments
├── data/
│   ├── raw/                        # Raw comment data
│   ├── processed/                  # Preprocessed data
│   └── slang_words.json            # Slang expansion dictionary
├── models/
│   └── distillbert_finetuned_v2/                # Fine-tuned model weights
├── src/
│   ├── components/
│   │   └── data_preprocessing.py   # TextPreprocessor class
│   ├── pipeline/
│   │   └── training_pipeline.py    # Model training pipeline
│   └── constants.py                # Configuration constants
└── requirements.txt                # Python dependencies

```


# Installation 🚀
## Prerequisites
- Python 3.8+
- CUDA 12.1+ (optional, for GPU acceleration)
- YouTube API Key

# ⚙️ Installation & Setup
**1) Clone the repository**

```bash
 git clone https://github.com/vineethdhagey/Video-Recommendation-Engine.git
 cd Video-Recommendation-Engine
```
**2) Create virtual environment**

```bash
python -m venv svre-env
   source svre-env/bin/activate  # On Windows: svre-env\Scripts\activate
```
**3) Install dependencies**
```bash
pip install -r requirements.txt
```
**4) Configure API Key**
- Get your YouTube API key from Google Cloud Console
- Update YOUTUBE_API_KEY in src/constants.py or app.py

**5) Download Pre-trained Model**
- Place the fine-tuned DistilBERT model in models/distillbert/
- Or train a new model using your own new dataset related to educational videos comments with labels using demo.py

# Usage 🎬
## Web Application
```bash
python app.py

```
Access the application at http://localhost:6969

Paste YouTube video URLs (one per line)
Click "Analyze & Rank"
View sentiment scores and rankings

# Batch Processing
```bash
python test.py
```
Processes multiple videos defined in the video_urls list and outputs rankings.

# Model Training
```bash
python demo.py
```
Trains a new DistilBERT model on your comment dataset.
# Sentiment Scoring 📊
**Score Range:** -1 to 1

- **Positive (1):** Favorable comments (praise, gratitude, appreciation)
- **Neutral (0):** Questions, factual observations, requests.
- **Negative (-1):** Criticism, complaints, dissatisfaction.

  **Confidence:** Percentage of comments with clear sentiment signals (|score| > 0.1).

  # Hybrid Model Details

  The application uses a weighted ensemble approach:

- **DistilBERT (70% weight):** Fine-tuned classification model trained on comment data
- **VADER (30% weight):** Lexicon-based sentiment analyzer for supplementary analysis
- **Combined Score:** (distilbert_score × 0.7) + (vader_score × 0.3)

This hybrid approach captures both semantic understanding (DistilBERT) and lexicon-based patterns (VADER).

# API Response Example
```bash
{
  "results": [
    {
      "url": "https://youtube.com/watch?v=...",
      "title": "Video abc123",
      "score": 0.425,
      "confidence": 87.3
    }
  ]
}
```
# Configuration ⚙️
Edit src/constants.py to customize:

```bash
YOUTUBE_API_KEY = "your_api_key_here"
DISTILBERT_WEIGHT = 0.7
VADER_WEIGHT = 0.3
MAX_COMMENTS = 1000
MODEL_SAVE_PATH = "models/distillbert"
SLANG_FILE = "data/slang_words.json"
EMOTICONS_FILE = "data/emoticons.json"
```



