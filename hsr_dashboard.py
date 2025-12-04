import streamlit as st
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import os

st.set_page_config(page_title="HSR Sentiment Dashboard", layout="wide", page_icon="images/matrix.ico")

# Load custom font
def load_custom_font():
    with open("css/PF Din Text Universal Medium.ttf", "rb") as f:
        font_data = base64.b64encode(f.read()).decode()
    return font_data

font_base64 = load_custom_font()

# Custom CSS
st.markdown(f"""
<style>
    @font-face {{
        font-family: 'PF Din Text';
        src: url(data:font/truetype;charset=utf-8;base64,{font_base64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}

    .main-header {{
        font-family: 'PF Din Text', sans-serif !important;
        font-size: 48px !important;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.2;
    }}
    .metric-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }}

    /* Apply custom font to other elements */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'PF Din Text', sans-serif !important;
    }}

    .stMetric {{
        font-family: 'PF Din Text', sans-serif !important;
    }}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    if not os.path.exists('reviews_with_versions.csv'):
        st.error("Data file not found! Please run the Jupyter notebook first.")
        return None
    df = pd.read_csv('reviews_with_versions.csv')
    df['review_dt'] = pd.to_datetime(df['review_date'])
    df['has_response'] = df['developer_response'].notna()

    # Recalculate version assignment with correct dates
    version_starts = {
        '1.0': '2023-04-26', '1.1': '2023-06-07', '1.2': '2023-07-19', '1.3': '2023-08-30',
        '1.4': '2023-10-11', '1.5': '2023-11-15', '1.6': '2023-12-27', '2.0': '2024-02-06',
        '2.1': '2024-03-27', '2.2': '2024-05-08', '2.3': '2024-06-19', '2.4': '2024-07-31',
        '2.5': '2024-09-10', '2.6': '2024-10-23', '2.7': '2024-12-04', '3.0': '2025-01-15',
        '3.1': '2025-02-26', '3.2': '2025-04-09', '3.3': '2025-05-21', '3.4': '2025-07-02',
        '3.5': '2025-08-13'
    }

    version_starts = {ver: pd.Timestamp(date) for ver, date in version_starts.items()}
    sorted_versions = sorted(version_starts.items(), key=lambda x: x[1])

    def assign_version(date):
        version = None
        for ver, start in sorted_versions:
            if date >= start:
                version = ver
            else:
                break
        return version

    df['version'] = df['review_dt'].apply(assign_version)

    return df

df = load_data()

if df is None:
    st.stop()

# Header
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Then use it:
img_base64 = get_base64_image("images/matrix.png")
st.markdown(
    f"""
    <p class="main-header">
        <img src="data:image/png;base64,{img_base64}" width="60" style="vertical-align: middle; margin-right: 8px;">
        Honkai: Star Rail Sentiment Dashboard
    </p>
    """,
    unsafe_allow_html=True
)

# Version metadata with banner images
# Using direct imgur links (more reliable than Wikia for hotlinking)
VERSION_DATA = {
    '1.0': {'name': 'The Rail Unto the Stars', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_1.0?file=Version_1.0_Splash_Screen.png'},
    '1.1': {'name': 'Galactic Roaming', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_1.1?file=Version_1.1_Splash_Screen.png'},
    '1.2': {'name': 'Even Immortality Ends', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_1.2?file=Version_1.2_Splash_Screen.png'},
    '1.3': {'name': 'Celestial Eyes Above Mortal Ruins', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_1.3?file=Version_1.3_Splash_Screen.png'},
    '1.4': {'name': 'Jolted Awake From a Winter Dream', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_1.4?file=Version_1.4_Splash_Screen.png'},
    '1.5': {'name': 'The Crepuscule Zone', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_1.5?file=Version_1.5_Splash_Screen.png'},
    '1.6': {'name': 'Crown of the Mundane and Divine', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_1.6?file=Version_1.6_Splash_Screen.png'},
    '2.0': {'name': 'If One Dreams At Midnight', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_2.0?file=Version_2.0_Splash_Screen.png'},
    '2.1': {'name': 'Into the Yawning Chasm', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_2.1?file=Version_2.1_Splash_Screen.png'},
    '2.2': {'name': 'Then Wake to Weep', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_2.2?file=Version_2.2_Splash_Screen.png'},
    '2.3': {'name': 'Farewell, Penacony', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_2.3?file=Version_2.3_Splash_Screen.png'},
    '2.4': {'name': 'Finest Duel Under the Pristine Blue', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_2.4?file=Version_2.4_Splash_Screen.png'},
    '2.5': {'name': 'Flying Aureus Shot to Lupine Rue', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_2.5?file=Version_2.5_Splash_Screen.png'},
    '2.6': {'name': 'Annals of Pinecany\'s Mappou Age', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_2.6?file=Version_2.6_Splash_Screen.png'},
    '2.7': {'name': 'A New Venture on the Eighth Dawn', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_2.7?file=Version_2.7_Splash_Screen.png'},
    '3.0': {'name': 'Paean of Era Nova', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_3.0?file=Version_3.0_Splash_Screen.png'},
    '3.1': {'name': 'Light Slips the Gate, Shadow Greets the Throne', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_3.1?file=Version_3.1_Splash_Screen.png'},
    '3.2': {'name': 'Through the Petals in the Land of Repose', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_3.2?file=Version_3.2_Splash_Screen.png'},
    '3.3': {'name': 'The Fall at Dawn\'s Rise', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_3.3?file=Version_3.3_Splash_Screen.png'},
    '3.4': {'name': 'For the Sun is Set to Die', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_3.4?file=Version_3.4_Splash_Screen.png'},
    '3.5': {'name': 'Before Their Deaths', 'banner': 'https://honkai-star-rail.fandom.com/wiki/Version_3.5?file=Version_3.5_Splash_Screen.png'},
}

# Check if local images exist
IMAGES_DIR = 'version_images'
USE_LOCAL_IMAGES = os.path.exists(IMAGES_DIR)

# Sidebar filters
st.sidebar.header("🔍 Filters")
sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    options=['POSITIVE', 'NEGATIVE'],
    default=['POSITIVE', 'NEGATIVE']
)

version_filter = st.sidebar.multiselect(
    "Game Version",
    options=sorted(df['version'].dropna().unique()),
    default=[]
)

# Show version info in sidebar
if version_filter:
    if len(version_filter) == 1 and version_filter[0] in VERSION_DATA:
        st.sidebar.info(f"🎮 **{VERSION_DATA[version_filter[0]]['name']}**")
    else:
        st.sidebar.info(f"📊 Showing {len(version_filter)} versions")
else:
    st.sidebar.success("📊 Showing all versions (1.0 - 3.5)")

rating_filter = st.sidebar.slider(
    "Rating Range",
    min_value=1,
    max_value=5,
    value=(1, 5)
)

# Apply filters
filtered_df = df[df['sentiment'].isin(sentiment_filter)]
if version_filter:
    filtered_df = filtered_df[filtered_df['version'].isin(version_filter)]
filtered_df = filtered_df[
    (filtered_df['rating'] >= rating_filter[0]) &
    (filtered_df['rating'] <= rating_filter[1])
]

# Show image status in sidebar
if USE_LOCAL_IMAGES:
    image_count = len([f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')])
    st.sidebar.success(f"🖼️ Images: {image_count}/21 loaded")
else:
    st.sidebar.warning("🖼️ Images: Placeholders only")
    if st.sidebar.button("📥 Download Images"):
        st.sidebar.info("Run: `python download_version_images.py`")

# Display version banner if single version selected
if version_filter and len(version_filter) == 1:
    selected_version = version_filter[0]
    if selected_version in VERSION_DATA:
        version_info = VERSION_DATA[selected_version]

        st.markdown("---")
        col_banner_left, col_banner_center, col_banner_right = st.columns([1, 3, 1])

        with col_banner_center:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;'>
                    <h2 style='color: white; margin: 0;'>Version {selected_version}</h2>
                    <h3 style='color: #f0f0f0; margin: 10px 0; font-weight: normal;'>{version_info['name']}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display banner image or styled placeholder
            if USE_LOCAL_IMAGES and os.path.exists(f"{IMAGES_DIR}/v{selected_version}.png"):
                st.image(f"{IMAGES_DIR}/v{selected_version}.png")
            else:
                # Styled placeholder
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
                                height: 300px; border-radius: 10px; display: flex;
                                align-items: center; justify-content: center; color: white;
                                font-size: 1.5em; text-align: center; padding: 20px;'>
                        <div>
                            <div style='font-size: 2em; margin-bottom: 10px;'>⭐</div>
                            <div style='font-weight: bold;'>Version {selected_version}</div>
                            <div style='font-size: 0.7em; margin-top: 10px; opacity: 0.9;'>{version_info['name']}</div>
                            <div style='font-size: 0.5em; margin-top: 15px; opacity: 0.7;'>
                                Splash art placeholder<br/>
                                <a href='https://honkai-star-rail.fandom.com/wiki/Version_{selected_version}'
                                   target='_blank' style='color: #fbbf24;'>View on Wiki →</a>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")

# Key Metrics
st.header("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Reviews",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
    )

with col2:
    positive_pct = (filtered_df['sentiment'] == 'POSITIVE').mean() * 100
    st.metric(
        label="Positive Sentiment",
        value=f"{positive_pct:.1f}%"
    )

with col3:
    avg_rating = filtered_df['rating'].mean()
    st.metric(
        label="Average Rating",
        value=f"{avg_rating:.2f}/5.0"
    )

with col4:
    response_rate = filtered_df['has_response'].mean() * 100
    st.metric(
        label="Response Rate",
        value=f"{response_rate:.1f}%"
    )

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "🎯 Aspect Analysis",
    "💬 Developer Responses",
    "🔍 Search",
    "📥 Data Explorer",
    "🎮 Version Gallery"
])

# TAB 1: Overview
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = filtered_df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rating Distribution")
        rating_counts = filtered_df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            color=rating_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment Trend Over Time")
    sentiment_over_time = filtered_df.groupby(filtered_df['review_dt'].dt.to_period('M'))['sentiment'].apply(
        lambda x: (x == 'POSITIVE').mean() * 100
    ).reset_index()
    sentiment_over_time['review_dt'] = sentiment_over_time['review_dt'].astype(str)

    fig = px.line(
        sentiment_over_time,
        x='review_dt',
        y='sentiment',
        labels={'review_dt': 'Month', 'sentiment': 'Positive Sentiment (%)'},
        markers=True
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% baseline")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment by Version")
    version_sentiment = filtered_df.groupby('version').agg({
        'sentiment': lambda x: (x == 'POSITIVE').mean() * 100,
        'review_dt': 'count'
    }).reset_index()
    version_sentiment.columns = ['version', 'positive_pct', 'count']
    version_sentiment = version_sentiment[version_sentiment['count'] > 50]

    fig = px.bar(
        version_sentiment,
        x='version',
        y='positive_pct',
        labels={'version': 'Version', 'positive_pct': 'Positive Sentiment (%)'},
        color='positive_pct',
        color_continuous_scale='RdYlGn'
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Aspect Analysis
with tab2:
    st.subheader("🎯 Aspect-Based Sentiment Analysis")
    st.write("Analyze sentiment for specific game aspects")

    aspects = {
        'Story': ['story', 'plot', 'narrative', 'quest', 'lore'],
        'Characters': ['character', 'waifu', 'husbando', 'design'],
        'Gacha': ['gacha', 'pull', 'pity', 'rate', 'summon'],
        'Gameplay': ['combat', 'gameplay', 'mechanic', 'battle', 'turn'],
        'Graphics': ['graphic', 'art', 'visual', 'animation']
    }

    aspect_results = []
    for aspect_name, keywords in aspects.items():
        pattern = '|'.join(keywords)
        aspect_reviews = filtered_df[filtered_df['review_description'].str.contains(pattern, case=False, na=False)]

        if len(aspect_reviews) > 0:
            positive_pct = (aspect_reviews['sentiment'] == 'POSITIVE').mean() * 100
            avg_rating = aspect_reviews['rating'].mean()
            count = len(aspect_reviews)

            aspect_results.append({
                'Aspect': aspect_name,
                'Reviews': count,
                'Positive %': positive_pct,
                'Avg Rating': avg_rating
            })

    aspect_df = pd.DataFrame(aspect_results)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Aspect Sentiment Breakdown**")
        fig = px.bar(
            aspect_df,
            x='Aspect',
            y='Positive %',
            color='Positive %',
            color_continuous_scale='RdYlGn',
            text='Positive %'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Review Count by Aspect**")
        fig = px.bar(
            aspect_df,
            x='Aspect',
            y='Reviews',
            color='Reviews',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("**Detailed Aspect Statistics**")
    st.dataframe(
        aspect_df.style.background_gradient(subset=['Positive %'], cmap='RdYlGn', vmin=0, vmax=100)
                       .format({'Positive %': '{:.1f}%', 'Avg Rating': '{:.2f}'}),
        use_container_width=True
    )

# TAB 3: Developer Responses
with tab3:
    st.subheader("💬 Developer Response Analysis")

    responded = df[df['has_response']]
    not_responded = df[~df['has_response']]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Responses",
            f"{len(responded):,}",
            f"{(len(responded)/len(df)*100):.1f}% of reviews"
        )

    with col2:
        response_to_negative = responded[responded['sentiment'] == 'NEGATIVE']
        st.metric(
            "Responses to Negative",
            f"{len(response_to_negative):,}",
            f"{(len(response_to_negative)/len(responded)*100):.1f}%"
        )

    with col3:
        avg_rating_with_response = responded['rating'].mean()
        avg_rating_without = not_responded['rating'].mean()
        st.metric(
            "Avg Rating (Responded)",
            f"{avg_rating_with_response:.2f}",
            f"{avg_rating_with_response - avg_rating_without:+.2f}"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Response Rate by Sentiment**")
        response_by_sentiment = df.groupby('sentiment')['has_response'].apply(
            lambda x: x.mean() * 100
        ).reset_index()
        response_by_sentiment.columns = ['Sentiment', 'Response Rate (%)']

        fig = px.bar(
            response_by_sentiment,
            x='Sentiment',
            y='Response Rate (%)',
            color='Sentiment',
            color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Response Rate by Rating**")
        response_by_rating = df.groupby('rating')['has_response'].apply(
            lambda x: x.mean() * 100
        ).reset_index()
        response_by_rating.columns = ['Rating', 'Response Rate (%)']

        fig = px.line(
            response_by_rating,
            x='Rating',
            y='Response Rate (%)',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("**Response Rate Over Time**")
    response_over_time = df.groupby(df['review_dt'].dt.to_period('M')).agg({
        'has_response': lambda x: x.mean() * 100,
        'rating': 'count'  # Use 'rating' instead of 'review_dt' for counting
    }).reset_index()
    response_over_time['review_dt'] = response_over_time['review_dt'].astype(str)
    response_over_time.columns = ['Month', 'Response Rate (%)', 'Total Reviews']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=response_over_time['Month'],
        y=response_over_time['Response Rate (%)'],
        mode='lines+markers',
        name='Response Rate (%)',
        yaxis='y'
    ))
    fig.add_trace(go.Bar(
        x=response_over_time['Month'],
        y=response_over_time['Total Reviews'],
        name='Total Reviews',
        yaxis='y2',
        opacity=0.3
    ))
    fig.update_layout(
        yaxis=dict(title='Response Rate (%)'),
        yaxis2=dict(title='Total Reviews', overlaying='y', side='right'),
        xaxis=dict(title='Month'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Sample Developer Responses**")
    sample_responses = responded[['review_description', 'rating', 'sentiment', 'developer_response']].sample(min(5, len(responded)))
    for idx, row in sample_responses.iterrows():
        with st.expander(f"⭐ {row['rating']}/5 - {row['sentiment']}"):
            st.write("**User Review:**")
            st.write(row['review_description'][:300] + "..." if len(row['review_description']) > 300 else row['review_description'])
            st.write("**Developer Response:**")
            st.info(row['developer_response'][:300] + "..." if len(str(row['developer_response'])) > 300 else row['developer_response'])

# TAB 4: Search
with tab4:
    st.subheader("🔍 Semantic Search")
    st.write("Search reviews using natural language queries")

    query = st.text_input("Enter your search query:", placeholder="e.g., character design, gacha rates, story")
    num_results = st.slider("Number of results:", 1, 20, 5)

    if query:
        try:
            from sentence_transformers import SentenceTransformer, util
            import torch

            @st.cache_resource
            def load_model():
                return SentenceTransformer('all-MiniLM-L6-v2')

            @st.cache_data
            def load_embeddings():
                return torch.load('review_embeddings.pt')

            model = load_model()
            embeddings = load_embeddings()

            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, embeddings)[0]
            top_indices = torch.topk(similarities, k=num_results).indices
            top_scores = torch.topk(similarities, k=num_results).values

            # Use original df, not filtered_df, because embeddings match original df indices
            results = df.iloc[top_indices.cpu().numpy()].copy()
            results['similarity'] = top_scores.cpu().numpy()

            # Now apply filters to the search results
            results = results[results['sentiment'].isin(sentiment_filter)]
            if version_filter:
                results = results[results['version'].isin(version_filter)]
            results = results[
                (results['rating'] >= rating_filter[0]) &
                (results['rating'] <= rating_filter[1])
            ]

            if len(results) > 0:
                st.success(f"Found {len(results)} results with avg similarity: {results['similarity'].mean():.3f}")
            else:
                st.warning("No results match your filters. Try adjusting the sidebar filters.")
                st.stop()

            for idx, row in results.iterrows():
                similarity_color = "🟢" if row['similarity'] > 0.7 else "🟡" if row['similarity'] > 0.5 else "🔴"
                sentiment_emoji = "😊" if row['sentiment'] == 'POSITIVE' else "😞"

                with st.expander(f"{similarity_color} Similarity: {row['similarity']:.3f} | ⭐ {row['rating']}/5 {sentiment_emoji}"):
                    st.write(row['review_description'])
                    st.caption(f"Version: {row['version']} | Date: {row['review_date']}")

        except FileNotFoundError:
            st.error("❌ Embeddings file not found!")
            st.info("📝 Please run the Jupyter notebook first to generate `review_embeddings.pt`")
            st.code("# In your notebook, run the cell that generates embeddings", language="python")
        except Exception as e:
            st.error(f"❌ Search error: {str(e)}")
            st.info("Make sure you have run the notebook and generated `review_embeddings.pt` file.")

# TAB 5: Data Explorer
with tab5:
    st.subheader("📥 Data Explorer")

    st.write("**Filter and export review data**")

    cols_to_show = st.multiselect(
        "Select columns to display:",
        options=filtered_df.columns.tolist(),
        default=['review_description', 'rating', 'sentiment', 'version', 'review_date']
    )

    st.dataframe(filtered_df[cols_to_show], use_container_width=True, height=400)

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = filtered_df[cols_to_show].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name='filtered_reviews.csv',
            mime='text/csv'
        )

    with col2:
        st.metric("Filtered Reviews", f"{len(filtered_df):,}")

    with col3:
        st.metric("Columns Selected", len(cols_to_show))

    st.write("**Dataset Statistics**")
    st.write(filtered_df[['rating', 'confidence']].describe())

# TAB 6: Version Gallery
with tab6:
    st.subheader("🎮 Version Gallery - All Patches")
    st.write("Explore all Honkai: Star Rail game versions and their splash arts")

    # Help notice for images
    if not USE_LOCAL_IMAGES:
        with st.expander("💡 How to Display Splash Art Images"):
            st.write("""
            **Currently showing:** Styled placeholders (images not loaded)

            **To display actual splash screen images:**
            1. Run the image downloader script:
               ```bash
               python download_version_images.py
               ```
            2. This will download all 21 version banners to `version_images/` folder
            3. Restart the dashboard to see the images

            **Or manually download:**
            - Visit: https://honkai-star-rail.fandom.com/wiki/Versions
            - Save images as `version_images/v{version}.png` (e.g., `v2.0.png`)

            **Note:** Placeholders show version info and link to wiki images.
            """)

    # Add filter for era
    era_filter = st.radio(
        "Filter by Era:",
        ["All", "Version 1.x", "Version 2.x", "Version 3.x"],
        horizontal=True
    )

    # Filter versions based on era
    if era_filter == "Version 1.x":
        display_versions = [v for v in VERSION_DATA.keys() if v.startswith('1.')]
    elif era_filter == "Version 2.x":
        display_versions = [v for v in VERSION_DATA.keys() if v.startswith('2.')]
    elif era_filter == "Version 3.x":
        display_versions = [v for v in VERSION_DATA.keys() if v.startswith('3.')]
    else:
        display_versions = list(VERSION_DATA.keys())

    # Display in grid format (3 columns)
    versions_per_row = 3
    for i in range(0, len(display_versions), versions_per_row):
        cols = st.columns(versions_per_row)
        for j, col in enumerate(cols):
            if i + j < len(display_versions):
                version = display_versions[i + j]
                version_info = VERSION_DATA[version]

                with col:
                    # Version card with stats
                    version_reviews = df[df['version'] == version]
                    review_count = len(version_reviews)
                    avg_sentiment = (version_reviews['sentiment'] == 'POSITIVE').mean() * 100 if len(version_reviews) > 0 else 0
                    avg_rating = version_reviews['rating'].mean() if len(version_reviews) > 0 else 0

                    st.markdown(
                        f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 15px; border-radius: 10px; margin-bottom: 10px; text-align: center;'>
                            <h3 style='color: white; margin: 0;'>Version {version}</h3>
                            <p style='color: #f0f0f0; margin: 5px 0; font-size: 0.9em;'>{version_info['name']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display banner or styled placeholder
                    if USE_LOCAL_IMAGES and os.path.exists(f"{IMAGES_DIR}/v{version}.png"):
                        st.image(f"{IMAGES_DIR}/v{version}.png")
                    else:
                        # Styled placeholder for gallery
                        st.markdown(
                            f"""
                            <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
                                        height: 200px; border-radius: 10px; display: flex;
                                        align-items: center; justify-content: center; color: white;
                                        font-size: 1em; text-align: center; padding: 10px;'>
                                <div>
                                    <div style='font-size: 2.5em;'>⭐</div>
                                    <div style='font-size: 0.6em; opacity: 0.8; margin-top: 5px;'>
                                        <a href='https://honkai-star-rail.fandom.com/wiki/Version_{version}'
                                           target='_blank' style='color: #fbbf24; text-decoration: none;'>
                                           View Splash Art →
                                        </a>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Stats
                    st.markdown(f"**Reviews:** {review_count:,}")
                    st.markdown(f"**Avg Rating:** {avg_rating:.2f}/5.0")
                    st.markdown(f"**Positive:** {avg_sentiment:.1f}%")

                    # Quick filter button
                    if st.button(f"View v{version} Details", key=f"btn_{version}"):
                        st.info(f"💡 Tip: Use the sidebar to filter by Version {version}")

    st.markdown("---")
    st.write("**Version Timeline:**")

    # Create a nice timeline view
    timeline_data = []
    for version, info in VERSION_DATA.items():
        version_reviews = df[df['version'] == version]
        timeline_data.append({
            'Version': version,
            'Name': info['name'],
            'Reviews': len(version_reviews),
            'Avg Rating': version_reviews['rating'].mean() if len(version_reviews) > 0 else 0,
            'Positive %': (version_reviews['sentiment'] == 'POSITIVE').mean() * 100 if len(version_reviews) > 0 else 0
        })

    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(
        timeline_df.style.background_gradient(subset=['Positive %'], cmap='RdYlGn', vmin=0, vmax=100)
                        .format({'Avg Rating': '{:.2f}', 'Positive %': '{:.1f}%', 'Reviews': '{:,}'}),
        use_container_width=True,
        height=600
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d;'>
    <p>Honkai: Star Rail Sentiment Analysis Dashboard</p>
    <p>CS410 Text Information Systems | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
