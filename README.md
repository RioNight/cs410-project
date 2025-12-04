# Honkai: Star Rail Sentiment Analysis
## CS410 Text Information Systems - Course Project

Comprehensive sentiment analysis system for 49,223 Honkai: Star Rail reviews using DistilBERT, aspect-based analysis, and interactive web dashboard.

**Live Dashboard:** [matrix-of-prescience-ultima.streamlit.app](https://matrix-of-prescience-ultima.streamlit.app)

---

## Table of Contents
1. [Features](#features)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage](#usage)
6. [Implementation](#implementation)
7. [Results](#results)

---

## Features

- **Sentiment Classification** - DistilBERT transformer (97.3% avg confidence)
- **Aspect Analysis** - Story, Characters, Gacha, Gameplay, Graphics
- **Developer Response Analysis** - Engagement patterns and impact
- **Semantic Search** - Natural language queries with sentence transformers
- **ML Comparison** - Naive Bayes, Logistic Regression vs DistilBERT
- **Interactive Dashboard** - Real-time filtering and exploration

---

## Dataset

### Collection

**Install scraper:**
```bash
pip install google-play-scraper
```

**Run collection script:**
```bash
python collect_reviews.py
```

**Details:**
- Source: Google Play Store API
- App ID: `com.HoYoverse.hkrpgoversea`
- Output: `reviews.csv` (~50K reviews)
- Runtime: 5-15 minutes

**CSV Columns:**
`review_id`, `user_name`, `review_description`, `rating`, `review_date`, `developer_response`, `developer_response_date`, `appVersion`, `thumbs_up`, `source`, `language_code`, `country_code`

---

## Installation

**Prerequisites:** Python 3.8+, 8GB RAM, 5GB disk space

**Install all dependencies:**
```bash
pip install pandas numpy scikit-learn transformers matplotlib seaborn wordcloud sentence-transformers torch streamlit plotly
```

**Or separately:**
```bash
# Notebook analysis
pip install pandas numpy scikit-learn transformers matplotlib seaborn wordcloud sentence-transformers torch

# Dashboard
pip install streamlit plotly
```

---

## Quick Start

### 1. Collect Data
```bash
pip install google-play-scraper
python collect_reviews.py
```

### 2. Run Analysis
```bash
jupyter notebook hsr_sentiment_analysis.ipynb
# Run all cells (10-30 min first time)
```

**Generates:**
- `reviews_with_sentiment.csv` - Sentiment labels
- `reviews_with_versions.csv` - Version assignments
- `review_embeddings.pt` - Search index
- `model_comparison.csv` - ML results
- `aspect_sentiment_analysis.csv` - Aspect data
- `developer_response_stats.csv` - Response metrics
- Multiple PNG visualizations

### 3. Launch Dashboard
```bash
streamlit run hsr_dashboard.py
```

Opens at `http://localhost:8501`

---

## Usage

### Jupyter Notebook

**Section 1-2:** Data loading, sentiment analysis, visualizations
**Section 3:** ML model comparison (Naive Bayes, Logistic Regression)
**Section 4:** Semantic search setup
**Section 5:** Aspect-based analysis
**Section 6:** Developer response analysis

### Dashboard

**Tabs:**
- **Overview** - Metrics, distributions, trends
- **Aspect Analysis** - Sentiment by game feature
- **Developer Responses** - Response rates and patterns
- **Search** - Semantic search with natural language
- **Data Explorer** - Filter and export data
- **Version Gallery** - Stats per game version (1.0-3.6)

**Sidebar Filters:** Sentiment, version, rating range

---

## Implementation

### Architecture

```
Data Pipeline:
reviews.csv → Cleaning → DistilBERT → Version Assignment → Cached Results

Analysis:
├── TF-IDF (5000 features, bigrams) → Naive Bayes / Logistic Regression
├── Aspect Keywords → Sentiment Aggregation
├── Response Metrics → Temporal Analysis
└── Sentence-BERT (384-dim) → Cosine Similarity Search

Visualization:
Matplotlib/Seaborn (static) + Plotly (interactive) + Streamlit (web UI)
```

### Key Techniques

**Sentiment Classification:**
- Model: DistilBERT (66M params, SST-2 pre-trained)
- Input: First 512 tokens
- Output: Label + confidence (0-1)
- Caching for efficiency

**Version Assignment:**
```python
version_starts = {'1.0': '2023-04-26', '1.1': '2023-06-07', ...}
def assign_version(date):
    for ver, start in sorted_versions:
        if date >= start: current_version = ver
        else: break
    return current_version
```

**Aspect Analysis:**
```python
aspects = {
    'Story': ['story', 'plot', 'narrative', 'quest', 'lore'],
    'Gacha': ['gacha', 'pull', 'pity', 'rate', 'summon'],
    # ...
}
# Regex pattern matching → sentiment aggregation
```

**Semantic Search:**
- Offline: Encode all reviews → save embeddings
- Online: Encode query → cosine similarity → top-K results
- Search time: <100ms for 49K reviews

**Dashboard:**
- Caching: `@st.cache_data` (load once), `@st.cache_resource` (models)
- Performance: Initial load ~2s, filter updates <100ms

---

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 81.5% | 80.7% | 86.1% | 83.3% |
| Logistic Regression | 82.3% | 81.2% | 87.3% | 84.1% |
| DistilBERT | 100% | 100% | 100% | 100% |

### Statistics

- **Total Reviews:** 49,223
- **Positive:** 53.6% | **Negative:** 46.4%
- **Average Rating:** 3.63/5.0
- **Sentiment-Rating Correlation:** 0.625
- **High Confidence (>0.95):** 88.7%

### Insights

**Aspect Rankings:**
1. Graphics: ~65% positive
2. Story: ~62% positive
3. Characters: ~58% positive
4. Gameplay: ~52% positive
5. Gacha: ~43% positive (most criticized)

**Developer Responses:**
- 8.8% overall response rate
- Higher response to 1-star reviews
- Lower response to 5-star reviews

---

## Troubleshooting

**"reviews.csv not found"**
```bash
python collect_reviews.py  # Collect data first
```

**"DistilBERT slow"**
- Expected: 10-30 min first run
- Use GPU for 10x speedup
- Results cached after first run

**"Dashboard won't start"**
```bash
pip install streamlit
ls reviews_with_versions.csv  # Check file exists
streamlit run hsr_dashboard.py --server.port 8502  # Try different port
```

**"Search not working"**
```bash
pip install sentence-transformers
# Re-run notebook cells 19-20
```

**"Out of memory"**
```python
df = df.sample(10000)  # Use smaller dataset for testing
```

---

## Project Structure

```
cs410-project/
├── README.md
├── collect_reviews.py                      # Data collection
├── hsr_sentiment_analysis.ipynb           # Main notebook
├── hsr_dashboard.py                        # Dashboard
├── css/PF Din Text Universal Medium.ttf   # Font
├── images/                                 # Logo, favicon
├── version_images/                         # Optional splash art
├── reviews.csv                             # Input data
├── reviews_with_sentiment.csv             # Output
├── reviews_with_versions.csv              # Output
├── review_embeddings.pt                   # Search index
├── model_comparison.csv                   # ML results
├── aspect_sentiment_analysis.csv          # Aspect data
├── developer_response_stats.csv           # Response metrics
└── *.png                                   # Visualizations
```

---

## Technologies

- [Transformers](https://huggingface.co/docs/transformers/) - DistilBERT
- [Sentence Transformers](https://www.sbert.net/) - Semantic search
- [Streamlit](https://streamlit.io/) - Dashboard
- [Plotly](https://plotly.com/python/) - Interactive charts
- [scikit-learn](https://scikit-learn.org/) - ML models
- [google-play-scraper](https://github.com/JoMingyu/google-play-scraper) - Data collection

**Course:** CS410 Text Information Systems, UIUC | **Semester:** Fall 2024

---

**Enjoy exploring Honkai: Star Rail sentiment analysis!** 🚀
