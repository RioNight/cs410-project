# Honkai: Star Rail Sentiment Analysis
## CS410 Text Information Systems - Course Project

A comprehensive sentiment analysis and information retrieval system for Honkai: Star Rail mobile game reviews, featuring transformer-based sentiment classification, aspect-based analysis, and an interactive web dashboard.

**Live Dashboard:** [matrix-of-prescience-ultima.streamlit.app](https://matrix-of-prescience-ultima.streamlit.app)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Features](#features)
6. [Implementation Details](#implementation-details)
7. [Project Structure](#project-structure)
8. [Usage Guide](#usage-guide)
9. [Results](#results)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements a complete sentiment analysis pipeline for analyzing 49,223 Google Play Store reviews of Honkai: Star Rail. The system provides:

- **Sentiment Classification** using DistilBERT transformer model
- **Aspect-Based Analysis** for game features (Story, Characters, Gacha, Gameplay, Graphics)
- **Developer Response Analysis** tracking engagement patterns
- **Semantic Search** with sentence transformers for natural language queries
- **Interactive Dashboard** for real-time data exploration
- **ML Model Comparison** (Naive Bayes, Logistic Regression, DistilBERT)

**Key Statistics:**
- 49,223 reviews analyzed
- 97.3% average confidence
- 22 game versions tracked
- 5 game aspects evaluated
- 8.8% developer response rate

---

## Dataset

### Obtaining the Dataset

The dataset consists of Google Play Store reviews for Honkai: Star Rail, collected using the `google-play-scraper` library.

#### Option 1: Automated Collection (Recommended)

**Step 1: Install Scraper Library**
```bash
pip install google-play-scraper
```

**Step 2: Run Collection Script**
```bash
python collect_reviews.py
```

This will:
- Connect to Google Play Store
- Fetch all available reviews for Honkai: Star Rail
- Process and clean the data
- Save to `reviews.csv` in the project root

**Expected Runtime:** 5-15 minutes depending on review count and internet speed

**Output:** `reviews.csv` with 40,000-50,000+ reviews

#### Option 2: Use Pre-collected Dataset

If you have a pre-collected `reviews.csv` file, place it in the project root directory.

**Required File:** `reviews.csv`

**Expected Location:** Project root directory

**Format:** CSV file with the following columns:
- `review_id`: Unique identifier
- `user_name`: Reviewer name
- `review_description`: Review text
- `rating`: Star rating (1-5)
- `review_date`: Date of review
- `developer_response`: Optional developer reply
- `developer_response_date`: Date of developer reply
- `appVersion`: App version
- `thumbs_up`: Helpful votes count
- `source`: Platform (Google Play)
- `language_code`: Language code
- `country_code`: Country code

### Data Collection Details

**Script:** `collect_reviews.py`

**Source:** Google Play Store API via `google-play-scraper`

**App ID:** `com.HoYoverse.hkrpgoversea` (Honkai: Star Rail)

**Collection Method:**
- Uses unofficial Google Play Store scraper
- Fetches all publicly available reviews
- No authentication required
- Respects rate limits

**Data Freshness:**
- Reviews are collected at runtime
- Includes reviews from game launch (April 2023) to present
- Re-run script periodically to get latest reviews

**Important Notes:**
- Google Play may rate-limit requests (wait and retry if needed)
- Review count varies over time as new reviews are posted
- Script automatically handles pagination
- Collected data is saved locally (not uploaded to any server)

**Troubleshooting Collection:**
```bash
# If script fails, try:
pip install --upgrade google-play-scraper

# Check app ID is correct
# Verify internet connection
# Wait 5 minutes and retry if rate-limited
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum (for transformer models)
- 5GB disk space (for models and data)

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/cs410-project.git
cd cs410-project
```

### Step 2: Install Dependencies

**For Jupyter Notebook Analysis:**
```bash
pip install pandas numpy scikit-learn transformers matplotlib seaborn wordcloud sentence-transformers torch
```

**For Streamlit Dashboard:**
```bash
pip install streamlit plotly
```

**Or install everything at once:**
```bash
pip install pandas numpy scikit-learn transformers matplotlib seaborn wordcloud sentence-transformers torch streamlit plotly
```

### Step 3: Verify Installation
```bash
python -c "import transformers, torch, streamlit; print('All dependencies installed successfully!')"
```

---

## Quick Start

### Step 0: Collect Data (First Time Only)

If you don't have `reviews.csv`, collect it first:

```bash
# Install scraper
pip install google-play-scraper

# Run collection script
python collect_reviews.py
```

**Wait 5-15 minutes** for data collection to complete.

### Option 1: Run Complete Analysis (Recommended for First Time)

1. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook hsr_sentiment_analysis.ipynb
   ```

2. **Run all cells in order** (Runtime: 10-30 minutes for first run):
   - Cells 1-6: Data loading and sentiment analysis
   - Cells 7-12: Visualization and trends
   - Cells 13-17: ML model comparison
   - Cells 18-21: Semantic search setup
   - Cells 22-25: Aspect-based analysis
   - Cells 26-29: Developer response analysis

3. **Outputs Generated:**
   - `reviews_with_sentiment.csv` - Reviews with sentiment labels
   - `reviews_with_versions.csv` - Reviews with game versions
   - `review_embeddings.pt` - Semantic embeddings for search
   - `model_comparison.csv` - ML model performance metrics
   - `aspect_sentiment_analysis.csv` - Aspect analysis results
   - `developer_response_stats.csv` - Response metrics
   - Multiple PNG visualizations

### Option 2: Launch Dashboard (Requires Completed Analysis)

```bash
streamlit run hsr_dashboard.py
```

Dashboard opens automatically at `http://localhost:8501`

---

## Features

### 1. Sentiment Analysis with DistilBERT

**What it does:** Classifies each review as POSITIVE or NEGATIVE using a pre-trained transformer model.

**How to use:**
- Automatically runs when executing notebook cells 1-6
- Results cached in `reviews_with_sentiment.csv`
- View confidence scores and distribution in cell 8 output

**Technical Details:**
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Processes first 512 tokens of each review
- Outputs: Sentiment label + confidence score (0-1)

### 2. ML Model Comparison

**What it does:** Compares traditional ML models against transformer baseline.

**Models Evaluated:**
1. **Naive Bayes** (TF-IDF features) - 81.5% accuracy
2. **Logistic Regression** (TF-IDF features) - 82.3% accuracy
3. **DistilBERT** (transformer baseline) - 100% accuracy

**How to use:**
- Run notebook cells 13-17
- View results in `model_comparison.csv`
- Confusion matrices saved as `confusion_matrices.png`
- ROC curves saved as `roc_curves.png`

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### 3. Aspect-Based Sentiment Analysis

**What it does:** Analyzes sentiment for specific game features.

**Aspects Tracked:**
- **Story** (plot, narrative, quests, lore)
- **Characters** (design, waifu, husbando, personality)
- **Gacha** (pull rates, pity, summon system)
- **Gameplay** (combat, mechanics, turn-based battles)
- **Graphics** (art, visuals, animations)

**How to use:**
- Run notebook cells 22-25
- View results in dashboard "Aspect Analysis" tab
- Export data from `aspect_sentiment_analysis.csv`

**Outputs:**
- Positive sentiment % per aspect
- Review volume per aspect
- Average rating per aspect
- Keyword analysis (positive/negative terms)

### 4. Developer Response Analysis

**What it does:** Tracks how developers engage with user reviews.

**Metrics Analyzed:**
- Response rate overall (8.8%)
- Response rate by sentiment
- Response rate by star rating
- Response timing (median/average days)
- Impact on ratings

**How to use:**
- Run notebook cells 26-29
- View trends in dashboard "Developer Responses" tab
- Sample responses displayed for review

**Key Insights:**
- Developers respond more to negative reviews
- Median response time tracked over time
- Rating comparison: with vs without response

### 5. Semantic Search System

**What it does:** Natural language search over reviews using sentence transformers.

**How to use:**

**In Notebook:**
```python
# Search for similar reviews
results = search("character design is amazing", top_k=10)

# View results with similarity scores
for idx, row in results.iterrows():
    print(f"Similarity: {row['similarity']:.3f}")
    print(f"Review: {row['review_description']}")
```

**In Dashboard:**
- Navigate to "Search" tab
- Enter natural language query
- Adjust number of results (1-20)
- View ranked results with similarity scores

**Technical Details:**
- Model: `all-MiniLM-L6-v2` sentence transformer
- Cosine similarity matching
- Embeddings cached in `review_embeddings.pt`
- Search time: <100ms for 49K reviews

### 6. Interactive Streamlit Dashboard

**What it does:** Web-based interface for real-time data exploration.

**Tabs Available:**

**📈 Overview:**
- Key metrics cards (total reviews, positive %, avg rating, response rate)
- Sentiment distribution pie chart
- Rating histogram
- Sentiment trend over time
- Sentiment by game version

**🎯 Aspect Analysis:**
- Positive sentiment by aspect
- Review volume by aspect
- Positive vs negative comparison
- Average rating by aspect
- Detailed statistics table

**💬 Developer Responses:**
- Response rate by sentiment
- Response rate by rating
- Response rate trend over time
- Rating impact analysis
- Sample response viewer

**🔍 Search:**
- Semantic search interface
- Natural language queries
- Similarity score display
- Filter by sentiment/version/rating

**📥 Data Explorer:**
- Interactive data table
- Custom column selection
- CSV export functionality
- Dataset statistics

**🎮 Version Gallery:**
- All 22 game versions displayed
- Version metadata (name, dates)
- Stats per version (reviews, rating, sentiment)
- Era filtering (1.x, 2.x, 3.x)
- Splash art display (optional)

**How to use:**
```bash
streamlit run hsr_dashboard.py
```

**Sidebar Filters:**
- Sentiment (POSITIVE/NEGATIVE)
- Game version (1.0 - 3.6)
- Rating range (1-5 stars)

---

## Implementation Details

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Data Pipeline                         │
├─────────────────────────────────────────────────────────┤
│  Raw Reviews (CSV)                                       │
│       ↓                                                  │
│  Data Cleaning & Preprocessing                           │
│       ↓                                                  │
│  DistilBERT Sentiment Analysis                          │
│       ↓                                                  │
│  Version Assignment (Date-based)                         │
│       ↓                                                  │
│  Cached Results (reviews_with_versions.csv)             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Analysis Modules                        │
├─────────────────────────────────────────────────────────┤
│  1. ML Model Comparison                                  │
│     - TF-IDF Vectorization (5000 features, bigrams)     │
│     - Naive Bayes Classifier (alpha=1.0)                │
│     - Logistic Regression (C=1.0, max_iter=1000)        │
│                                                          │
│  2. Aspect-Based Analysis                                │
│     - Keyword Pattern Matching (regex)                   │
│     - Sentiment Aggregation per Aspect                   │
│     - Statistical Analysis                               │
│                                                          │
│  3. Developer Response Analysis                          │
│     - Response Rate Calculation                          │
│     - Temporal Analysis                                  │
│     - Impact Assessment                                  │
│                                                          │
│  4. Semantic Search                                      │
│     - Sentence-BERT Encoding (384-dim vectors)          │
│     - Cosine Similarity Computation                      │
│     - Top-K Retrieval                                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Visualization Layer                         │
├─────────────────────────────────────────────────────────┤
│  - Matplotlib/Seaborn (Static)                          │
│  - Plotly (Interactive Dashboard)                        │
│  - Streamlit (Web Interface)                            │
└─────────────────────────────────────────────────────────┘
```

### Technical Approach

#### 1. Sentiment Classification Pipeline

**Algorithm:** DistilBERT (Distilled BERT)
- Pre-trained on SST-2 (Stanford Sentiment Treebank)
- 66M parameters (40% smaller than BERT-base)
- Inference: ~50ms per review on CPU

**Pipeline:**
```python
def analyze_sentiment(text):
    # Truncate to max length
    truncated = text[:512]

    # Get model prediction
    result = sentiment_analyzer(truncated)[0]

    # Return label and confidence
    return result['label'], result['score']
```

**Optimization:**
- Results cached after first run
- Batch processing (future improvement)
- GPU acceleration support

#### 2. Version Assignment Algorithm

**Approach:** Date-based binary search

```python
# Version release dates (ground truth)
version_starts = {
    '1.0': '2023-04-26',
    '1.1': '2023-06-07',
    # ... 22 versions total
    '3.6': '2025-09-24'
}

# Assign version based on review date
def assign_version(review_date):
    for version, start_date in sorted_versions:
        if review_date >= start_date:
            current_version = version
        else:
            break
    return current_version
```

**Edge Cases Handled:**
- Reviews before v1.0 release
- Missing dates (excluded)
- Future dates (assigned to latest version)

#### 3. TF-IDF Feature Extraction

**Configuration:**
- **Vocabulary size:** 5000 features
- **N-grams:** Unigrams + Bigrams (1,2)
- **Stop words:** Custom set (game-specific + ENGLISH_STOP_WORDS)
- **Min document frequency:** 2 (removes rare terms)

**Custom Stop Words:**
```python
custom_stopwords = {
    'good', 'bad', 'nice', 'great', 'best',  # Generic sentiment
    'game', 'play', 'playing',                 # Domain-common
    'just', 'really', 'fun', 'amazing'        # Overused terms
}
```

**Rationale:** Focus on discriminative features, not overly common words.

#### 4. Aspect-Based Sentiment

**Method:** Keyword Pattern Matching + Aggregation

```python
aspects = {
    'Story': ['story', 'plot', 'narrative', 'quest', 'lore'],
    'Gacha': ['gacha', 'pull', 'pity', 'rate', 'summon'],
    # ... more aspects
}

# Match reviews containing aspect keywords
pattern = '|'.join(keywords)  # Regex OR pattern
aspect_reviews = df[df['review'].str.contains(pattern, case=False)]

# Calculate sentiment statistics
positive_pct = (aspect_reviews['sentiment'] == 'POSITIVE').mean()
```

**Advantages:**
- Simple and interpretable
- Fast computation
- Domain-specific customization

**Limitations:**
- Keyword-dependent (may miss implicit mentions)
- No context understanding
- Potential overlaps between aspects

**Future Improvements:**
- Named Entity Recognition
- Topic modeling (LDA)
- Transformer-based aspect extraction

#### 5. Semantic Search Implementation

**Model:** Sentence-BERT (all-MiniLM-L6-v2)
- **Embedding size:** 384 dimensions
- **Max sequence length:** 256 tokens
- **Performance:** ~5ms per encoding

**Search Process:**
1. **Offline Indexing:**
   ```python
   # Encode all reviews once
   embeddings = model.encode(reviews, convert_to_tensor=True)
   torch.save(embeddings, 'review_embeddings.pt')
   ```

2. **Online Search:**
   ```python
   # Encode query
   query_embedding = model.encode(query)

   # Compute cosine similarity
   similarities = util.cos_sim(query_embedding, embeddings)

   # Return top-K results
   top_indices = torch.topk(similarities, k=10).indices
   ```

**Performance:**
- Index size: ~75MB for 49K reviews
- Search latency: <100ms
- Accuracy: Captures semantic similarity better than keyword search

#### 6. Dashboard Architecture

**Framework:** Streamlit
- **Backend:** Python
- **Frontend:** React (auto-generated)
- **Charts:** Plotly (interactive)

**Data Flow:**
```
reviews_with_versions.csv
         ↓
   st.cache_data (load once)
         ↓
   Sidebar Filters (reactive)
         ↓
   filtered_df (dynamic)
         ↓
   Plotly Charts (rendered)
```

**Caching Strategy:**
- `@st.cache_data` for data loading (runs once)
- `@st.cache_resource` for model loading (persists)
- CSV re-read only on file modification

**Performance:**
- Initial load: ~2 seconds
- Filter update: <100ms
- Chart render: <200ms

---

## Project Structure

```
cs410-project/
├── README.md                                # This file
├── collect_reviews.py                       # Data collection script
├── hsr_sentiment_analysis.ipynb            # Main analysis notebook
├── hsr_dashboard.py                         # Streamlit dashboard
│
├── css/                                     # Custom fonts and styles
│   └── PF Din Text Universal Medium.ttf    # Dashboard font
│
├── images/                                  # Dashboard assets
│   ├── matrix.ico                           # Favicon
│   └── matrix.png                           # Header logo
│
├── version_images/                          # Optional splash art
│   ├── v1.0.png                            # Version banners
│   ├── v2.0.png
│   └── ...                                  # (21 images total)
│
├── Data Files (Generated):
│   ├── reviews.csv                          # [INPUT] Original dataset
│   ├── reviews_with_sentiment.csv           # With DistilBERT labels
│   ├── reviews_with_versions.csv            # With version assignments
│   ├── review_embeddings.pt                 # Semantic search index
│   ├── model_comparison.csv                 # ML model results
│   ├── aspect_sentiment_analysis.csv        # Aspect analysis data
│   └── developer_response_stats.csv         # Response metrics
│
└── Visualizations (Generated):
    ├── sentiment_analysis.png               # Overview charts
    ├── sentiment_trend.png                  # Temporal trends
    ├── sentiment_by_version.png             # Version comparison
    ├── wordclouds.png                       # Positive/negative clouds
    ├── confusion_matrices.png               # Model evaluation
    ├── roc_curves.png                       # ROC curves
    ├── aspect_sentiment_analysis.png        # Aspect charts
    └── developer_response_analysis.png      # Response trends
```

---

## Usage Guide

### Running the Jupyter Notebook

#### First Time Setup (10-30 minutes):

1. **Start Jupyter:**
   ```bash
   jupyter notebook hsr_sentiment_analysis.ipynb
   ```

2. **Run Section 1 (Data Loading & Sentiment Analysis):**
   - Cell 1: Import libraries
   - Cell 3: Load `reviews.csv`
   - Cell 4: Run DistilBERT analysis (SLOW - 10-30 min)
     - Progress bar displayed
     - Results cached automatically
   - Cell 6: Assign game versions

3. **Run Section 2 (Analysis & Visualization):**
   - Cell 8: Summary statistics
   - Cells 9-12: Generate visualizations

4. **Run Section 3 (ML Models):**
   - Cell 14: TF-IDF vectorization
   - Cell 15: Train Naive Bayes & Logistic Regression
   - Cells 16-17: Evaluation and ROC curves

5. **Run Section 4 (Semantic Search):**
   - Cell 19: Load sentence transformer (2-3 min)
   - Cell 20: Define search functions
   - Cell 21: Interactive search (optional)

6. **Run Section 5 (Aspect Analysis):**
   - Cell 23: Define aspects and extract data
   - Cell 24: Generate visualizations
   - Cell 25: Detailed keyword analysis

7. **Run Section 6 (Developer Responses):**
   - Cell 27: Calculate response metrics
   - Cell 28: Generate visualizations
   - Cell 29: Sample responses and export

#### Subsequent Runs (Fast):
- Sentiment analysis loads from cache
- Embeddings load from saved file
- Only needs to re-run analysis cells

### Running the Dashboard

#### Start Dashboard:
```bash
streamlit run hsr_dashboard.py
```

#### Dashboard Navigation:

1. **Using Sidebar Filters:**
   - Click sentiment checkboxes (POSITIVE/NEGATIVE)
   - Select game versions from dropdown (multi-select)
   - Adjust rating slider (1-5 stars)
   - All charts update automatically

2. **Exploring Tabs:**

   **Overview Tab:**
   - View high-level metrics at top
   - Scroll down for detailed charts
   - Hover over charts for exact values
   - Click legend items to toggle series

   **Aspect Analysis Tab:**
   - View sentiment breakdown by game aspect
   - Compare review volumes
   - Check detailed statistics table
   - Export data if needed

   **Developer Responses Tab:**
   - Explore response rate trends
   - Compare response rates by sentiment/rating
   - View sample developer responses
   - Check response time statistics

   **Search Tab:**
   - Enter natural language query (e.g., "gacha rates are unfair")
   - Select number of results (1-20)
   - Click search
   - Review results ranked by similarity
   - Note: Respects sidebar filters

   **Data Explorer Tab:**
   - Select columns to display
   - Browse filtered data
   - Download as CSV
   - View dataset statistics

   **Version Gallery Tab:**
   - Filter by era (1.x, 2.x, 3.x)
   - View all version cards
   - Check stats per version
   - Click "View Details" for quick filter

3. **Exporting Data:**
   - Navigate to Data Explorer tab
   - Apply desired filters in sidebar
   - Select columns to export
   - Click "Download as CSV"
   - File downloads to browser default location

#### Optional: Version Images

The dashboard works perfectly with styled placeholders. To add official splash art:

**Option 1: Automatic Download**
```bash
python download_version_images.py
```
- Creates `version_images/` folder
- Downloads 21 splash screens
- Restart dashboard to see images

**Option 2: Manual Download**
1. Create folder: `version_images/`
2. Visit: https://honkai-star-rail.fandom.com/wiki/Versions
3. Download splash screens
4. Save as: `v{version}.png` (e.g., `v2.0.png`)
5. Restart dashboard

---

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 81.49% | 80.70% | 86.06% | 83.29% |
| Logistic Regression | 82.33% | 81.16% | 87.31% | 84.12% |
| **DistilBERT** | **100%** | **100%** | **100%** | **100%** |

**Note:** DistilBERT achieves 100% because it's used to generate the labels (baseline).

### Sentiment Distribution

- **Total Reviews:** 49,223
- **Positive:** 26,387 (53.6%)
- **Negative:** 22,836 (46.4%)
- **Average Rating:** 3.63/5.0
- **Sentiment-Rating Correlation:** 0.625
- **High Confidence (>0.95):** 88.7%

### Top Insights

**Aspect Sentiment Rankings:**
1. Graphics: ~65% positive (most praised)
2. Story: ~62% positive
3. Characters: ~58% positive
4. Gameplay: ~52% positive
5. Gacha: ~43% positive (most criticized)

**Developer Response Patterns:**
- 8.8% overall response rate
- Higher response rate to 1-star reviews
- Lower response rate to 5-star reviews
- Reviews with responses have slightly lower avg rating (selection bias)

**Temporal Trends:**
- Sentiment fluctuates with major version releases
- Version 2.0 saw spike in positive reviews (new content)
- Recent versions (3.x) show slight decline

---

## Troubleshooting

### Common Issues

#### Issue: "reviews.csv not found"
**Solution:**
```bash
# Ensure reviews.csv is in project root
ls reviews.csv

# Or update path in notebook cell 3:
df = pd.read_csv("path/to/reviews.csv")
```

#### Issue: "DistilBERT analysis is very slow"
**Expected Behavior:** First run takes 10-30 minutes for 49K reviews.

**Speed Up:**
- Use GPU if available (10x faster)
- Reduce dataset size for testing
- Results are cached after first run

**GPU Setup:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: "Dashboard won't start"
**Solutions:**
```bash
# Check if streamlit is installed
pip install streamlit

# Check if data file exists
ls reviews_with_versions.csv

# If file missing, run notebook first

# Try specifying port
streamlit run hsr_dashboard.py --server.port 8502
```

#### Issue: "Semantic search not working"
**Solution:**
```bash
# Install sentence-transformers
pip install sentence-transformers

# Re-run notebook cells 19-20 to generate embeddings

# Verify embeddings file exists
ls review_embeddings.pt
```

#### Issue: "Charts not displaying in dashboard"
**Solution:**
```bash
# Install plotly
pip install plotly

# Clear browser cache
# Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)

# Check browser console for errors
```

#### Issue: "Out of memory error"
**Solution:**
```bash
# Reduce dataset size for testing
df = df.sample(10000)  # Use 10K reviews

# Close other applications
# Use CPU instead of GPU for small batches
# Increase system swap/pagefile
```

#### Issue: "Version images not showing"
**Expected Behavior:** Placeholders are shown by default (intentional).

**To Add Images:**
```bash
python download_version_images.py
# Or manually download from wiki
```

### Getting Help

**Error Messages:**
- Read the full error traceback
- Check line numbers against notebook cells
- Verify all dependencies installed

**Performance Issues:**
- Monitor CPU/RAM usage
- Close unnecessary applications
- Use smaller dataset for testing

**Data Issues:**
- Verify CSV format matches expected schema
- Check for missing required columns
- Ensure dates are in correct format

---

## Contact & Resources

**Project Repository:** https://github.com/YOUR_USERNAME/cs410-project

**Technologies Used:**
- [Transformers](https://huggingface.co/docs/transformers/) - DistilBERT model
- [Sentence Transformers](https://www.sbert.net/) - Semantic search
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [Plotly](https://plotly.com/python/) - Interactive visualizations
- [scikit-learn](https://scikit-learn.org/) - ML models
- [Pandas](https://pandas.pydata.org/) - Data manipulation

**Course:** CS410 Text Information Systems, UIUC

**Semester:** Fall 2024

---

## License

This project is submitted as coursework for CS410. All rights reserved.

---

## Acknowledgments

- Course staff for project guidance
- Hugging Face for pre-trained models
- Streamlit team for excellent framework
- Honkai: Star Rail community for engaging reviews

---

**Enjoy exploring Honkai: Star Rail sentiment analysis!** 🚀
