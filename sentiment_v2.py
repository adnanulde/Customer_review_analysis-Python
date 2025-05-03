# Sentiment Analysis Project
# Import necessary libraries

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
import io
import base64

# Set page config at the very beginning
st.set_page_config(page_title="Sentiment Analysis App", layout="wide", page_icon="😊")

# Download necessary NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        # Check if NLTK resources are already downloaded
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        # If not, download them
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# Call the function to download resources
download_nltk_resources()

# Text preprocessing functions
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def remove_stopwords(text):
    try:
        stop_words = set(stopwords.words('english'))
        # Remove important negation words from stopwords
        stop_words.discard('no')
        stop_words.discard('not')
        stop_words.discard('nor')
        stop_words.discard('neither')
        stop_words.discard('never')
        words = text.split()
        return " ".join([word for word in words if word not in stop_words])
    except Exception as e:
        st.error(f"Error in stopwords removal: {e}")
        return text

def stem_text(text):
    try:
        stemmer = PorterStemmer()
        words = text.split()  # Simple split instead of word_tokenize
        return " ".join([stemmer.stem(word) for word in words])
    except Exception as e:
        st.error(f"Error in stemming: {e}")
        return text  # Return original text if there's an error

def lemmatize_text(text):
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()  # Simple split instead of word_tokenize
        return " ".join([lemmatizer.lemmatize(word) for word in words])
    except Exception as e:
        st.error(f"Error in lemmatization: {e}")
        return text  # Return original text if there's an error

def preprocess_text(text, remove_stops=True, stemming=False, lemmatization=True):
    text = clean_text(text)
    if remove_stops:
        text = remove_stopwords(text)
    if stemming:
        text = stem_text(text)
    if lemmatization:
        text = lemmatize_text(text)
    return text

def load_custom_data(file):
    try:
        df = pd.read_csv(file)
        
        # Check for CSV formatting issues
        if len(df.columns) == 1 and ',' in df.columns[0]:
            st.error("Your CSV file appears to have formatting issues. It might be using a different delimiter than expected.")
            st.info("Please make sure your CSV file is properly formatted with comma separators.")
            return None
        
        # Instead of showing column arrays, show a cleaner message
        st.info(f"Found {len(df.columns)} columns in your dataset: {', '.join(df.columns)}")
        
        # Verify required columns exist
        required_columns = ['review_text', 'sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.info("Your CSV should contain at least 'review_text' and 'sentiment' columns. The 'sentiment' column should contain values like 'positive' or 'negative'.")
            return None
            
        # Check if sentiment column has the right format - don't display arrays
        unique_sentiments = df['sentiment'].unique().tolist()
        if len(unique_sentiments) <= 5:  # Only show if there's a reasonable number
            st.info(f"Detected sentiment classes: {', '.join(str(s) for s in unique_sentiments)}")
        else:
            st.info(f"Detected {len(unique_sentiments)} different sentiment classes")
        
        if not all(sentiment in ['positive', 'negative'] for sentiment in unique_sentiments):
            st.warning("Your sentiment column contains values other than 'positive' and 'negative'. The model expects these exact values.")
            
            # Try to map common sentiment values
            if 'rating' in df.columns:
                st.info("Found 'rating' column. Converting ratings to sentiments (≥4 = positive, <4 = negative).")
                df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')
            else:
                # Let user map the values
                st.info("Please map your sentiment values to 'positive' and 'negative':")
                for val in unique_sentiments:
                    mapped_val = st.selectbox(f"Map '{val}' to:", ['positive', 'negative'], key=f"map_{val}")
                    df['sentiment'] = df['sentiment'].replace(val, mapped_val)
        
        # Preprocess the text
        st.info("Preprocessing text data...")
        df['processed_text'] = df['review_text'].fillna("").astype(str).apply(preprocess_text)
        
        st.success(f"Successfully loaded and processed {len(df)} reviews!")
        return df
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Show more details to help debug CSV issues
        st.error("Please check that your CSV file is properly formatted. Common issues include:")
        st.markdown("""
        - Incorrect delimiters (should use commas)
        - Missing or incorrectly named columns
        - Encoding issues (try saving as UTF-8)
        - Quoting issues with text that contains commas
        """)
        return None

# Load and preprocess the data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_amazon_data():
    # Load data from CSV file
    file_path = "amazon_reviews_dataset.csv"  # Relative path
    # Try to find the file in different possible locations
    possible_paths = [
        file_path,
        f"C:/Users/adnan/Python/Python_PROJECT/{file_path}",
        f"./{file_path}"
    ]
    
    try:
        # Try each possible path
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                print(f"Successfully loaded {len(df)} reviews from {path}")
                break
            except FileNotFoundError:
                continue
        else:  # If no file was found
            raise FileNotFoundError(f"Could not find {file_path} in any of the expected locations")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Falling back to default dataset")
        # Fallback to a small sample dataset for demonstration
        data = {
            'review_text': [
                # Positive examples
                "This product is amazing! I love it so much and would definitely recommend it to everyone.",
                "I bought this for my daughter and she absolutely loves it! Great quality and fast shipping.",
                "Very satisfied with this purchase, exceeded my expectations.",
                "Good value for money, would buy again.",
                "This is exactly what I was looking for. Perfect fit and great design.",
                "Excellent product! Works perfectly and was delivered on time.",
                "I've been using this for a month now and it's still working great. Very durable.",
                "The customer service was exceptional. They resolved my issue immediately.",
                "This product has improved my daily routine significantly. Highly recommend!",
                "Worth every penny! Quality construction and beautiful design.",
                "Easy to set up and use. The instruction manual was clear and helpful.",
                "I'm impressed with the attention to detail. You can tell they care about quality.",
                "My whole family loves this product. We use it every day.",
                "This is my second purchase of this item - that's how much I like it!",
                "The price point is perfect for the quality you receive.",
                "It arrived earlier than expected and in perfect condition.",
                "This product has exceeded my expectations in every way.",
                "I've tried many similar products, but this is by far the best.",
                "The design is both functional and attractive. Great engineering!",
                "Five stars isn't enough for how good this product is!",
                
                # Negative examples
                "Worst purchase ever. Poor quality and terrible customer service.",
                "Don't waste your money. It broke after two days of use.",
                "The product arrived damaged and the company refused to replace it.",
                "Disappointed with the quality, doesn't match the description at all.",
                "This is the worst product I've ever purchased.",
                "I hate this so much, complete waste of money.",
                "Very bad quality, broke immediately.",
                "Terrible experience, would give zero stars if possible.",
                "Absolutely awful, don't buy this.",
                "The product stopped working after a week. Complete junk.",
                "Misleading product description. Not at all what I expected.",
                "Poor craftsmanship. You can tell it's cheaply made.",
                "Save your money and look elsewhere. This isn't worth it.",
                "Had to return it immediately. Didn't work as advertised.",
                "The customer service was unhelpful when I reported issues.",
                "Overpriced for such poor quality. I regret this purchase.",
                "The design is flawed and makes it difficult to use.",
                "It looks nothing like the pictures online. Very disappointing.",
                "Broke during first use. Clearly not tested before shipping.",
                "Instructions were confusing and parts were missing.",
                "This product is dangerous and should be recalled.",
                "Extremely disappointed. Don't believe the positive reviews.",
                "The materials feel cheap and it's poorly constructed.",
                "Horrible product that fails to do what it promises.",
                "Waste of time and money. Avoid at all costs."
            ],
            'rating': [
                # Ratings for positive reviews
                5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 4, 5, 5, 5, 4, 5, 5, 5, 4, 5,
                # Ratings for negative reviews
                1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1
            ]
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
    
    # Create binary sentiment column based on rating if it doesn't exist
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')
    
    # Preprocess the text
    df['processed_text'] = df['review_text'].apply(preprocess_text)
    
    return df

# Train the model
@st.cache_resource
def train_model(df):
    # Prepare the data
    X = df['processed_text']
    y = df['sentiment']
    
    # Convert sentiment to numeric
    sentiment_map = {'negative': 0, 'positive': 1}
    y_numeric = y.map(sentiment_map)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric)
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train the model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, accuracy, sentiment_map

# Predict sentiment
def predict_sentiment(text, model, vectorizer, sentiment_map):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    sentiment_numeric = model.predict(vectorized_text)[0]
    
    # Convert numeric prediction back to text
    reverse_map = {v: k for k, v in sentiment_map.items()}
    sentiment = reverse_map[sentiment_numeric]
    
    # Get probability scores
    probabilities = model.predict_proba(vectorized_text)[0]
    
    return sentiment, probabilities, reverse_map

# Generate visualizations
def generate_wordcloud(df, sentiment):
    try:
        text = ' '.join(df[df['sentiment'] == sentiment]['processed_text'])
        if not text.strip():  # Check if text is empty
            st.warning(f"No text data available for {sentiment} sentiment")
            return None
        
        # Create and generate the wordcloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=100,
                            colormap='viridis' if sentiment == 'positive' else 'RdGy').generate(text)
        
        # Convert WordCloud directly to image
        return wordcloud.to_image()
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None

def plot_sentiment_distribution(df):
    try:
        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create the bar plot using seaborn
        sns.countplot(data=df, x='sentiment', ax=ax,
                     palette={'positive': 'green', 'negative': 'red'},
                     order=['positive', 'negative'])
        
        # Set titles and labels
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf
    except Exception as e:
        st.error(f"Error plotting sentiment distribution: {e}")
        plt.close('all')
        return None

# Process uploaded CSV file
# In the process_csv function, make these changes:
def process_csv(file, model, vectorizer, sentiment_map):
    try:
        df = pd.read_csv(file)
        
        # Instead of showing raw column array, show a cleaner message
        st.info(f"CSV loaded successfully with {len(df.columns)} columns: {', '.join(df.columns)}")
        
        # Check if required column exists with more flexible column matching
        text_column = None
        possible_columns = ['review', 'text', 'comment', 'review_text', 'reviews', 'comments', 'content']
        
        for col in possible_columns:
            if col in df.columns:
                text_column = col
                break
                
        # If no matching column found, let user select a column
        if text_column is None:
            if len(df.columns) > 0:
                text_column = st.selectbox("Please select the column containing review text:", df.columns)
            else:
                st.error("The CSV file appears to be empty or corrupted.")
                return None
        
        st.info(f"Using column '{text_column}' for sentiment analysis.")
        
        # Process each review, handling NaN values
        df['processed_text'] = df[text_column].fillna("").astype(str).apply(preprocess_text)
        
        # Predict sentiment for each review
        with st.spinner(f"Analyzing {len(df)} reviews..."):
            sentiments = []
            probabilities_pos = []
            
            for text in df['processed_text']:
                if text.strip():  # Check if text is not empty after processing
                    vectorized_text = vectorizer.transform([text])
                    sentiment_numeric = model.predict(vectorized_text)[0]
                    probs = model.predict_proba(vectorized_text)[0]
                    reverse_map = {v: k for k, v in sentiment_map.items()}
                    sentiment = reverse_map[sentiment_numeric]
                    pos_prob = probs[1] if sentiment_map['positive'] == 1 else probs[0]
                else:
                    sentiment = "unknown"  # For empty strings
                    pos_prob = 0.0
                    
                sentiments.append(sentiment)
                probabilities_pos.append(pos_prob)
            
            df['predicted_sentiment'] = sentiments
            df['confidence_score'] = probabilities_pos
        
        st.success(f"Analysis complete! Processed {len(df)} reviews.")
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        st.error("Please check that your CSV file is properly formatted and contains text data.")
        return None
# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .sentiment-positive {
        color: green;
        font-weight: bold;
    }
    .sentiment-negative {
        color: red;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0eaff;
        border-bottom: 2px solid #1e3a8a;
    }
    </style>
""", unsafe_allow_html=True)

try:
    # Add header first
    st.title("📊 Sentiment Analysis Application")
    st.write("Analyze the sentiment of text reviews or upload a CSV file for batch processing.")
    
    # Add an option to upload training data first
    st.sidebar.header("Model Training")
    train_file = st.sidebar.file_uploader("Upload training dataset (CSV)", type="csv", key="train_data")
    
    if train_file:
        st.sidebar.info("Using uploaded file for training")
        df = load_custom_data(train_file)
        if df is None:
            st.sidebar.error("Could not load custom dataset. Using default dataset instead.")
            df = load_amazon_data()
    else:
        st.sidebar.info("Using default dataset for training")
        df = load_amazon_data()
    
    # Train model with progress indicator
    train_status = st.sidebar.empty()
    train_status.info("Training model... This might take a moment.")
    model, vectorizer, accuracy, sentiment_map = train_model(df)
    train_status.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
    
    # Download trained model option
    if st.sidebar.button("Download Trained Model"):
        # Save model and vectorizer to bytes
        model_bytes = pickle.dumps((model, vectorizer, sentiment_map))
        # Create download button
        st.sidebar.download_button(
            label="Download Model Files",
            data=model_bytes,
            file_name="sentiment_model.pkl",
            mime="application/octet-stream"
        )
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Text Input", "📁 CSV Upload", "📈 Model Performance", "🔍 Data Exploration"])
    
    # Tab 1: Text Input
    with tab1:
        st.header("Analyze Text Sentiment")
        
        # Text input
        user_text = st.text_area("Enter text to analyze:", height=150)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
        
        with col2:
            st.write("")  # Spacer
        
        if analyze_btn and user_text:
            with st.spinner("Analyzing..."):
                sentiment, probabilities, reverse_map = predict_sentiment(user_text, model, vectorizer, sentiment_map)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Sentiment display
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color:white; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                        <h3>Sentiment</h3>
                        <p class="sentiment-{sentiment}" style="font-size:24px;">{sentiment.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Create a dictionary for probabilities
                    prob_dict = {reverse_map[i]: float(prob) for i, prob in enumerate(probabilities)}
                    
                    # Display probabilities as a horizontal bar chart
                    st.markdown('<h3>Confidence Scores</h3>', unsafe_allow_html=True)
                    
                    # Sort by probability values
                    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
                    
                    for sentiment, prob in sorted_probs.items():
                        # Choose color based on sentiment
                        if sentiment == 'positive':
                            color = 'green'
                        else:
                            color = 'red'
                        
                        # Display progress bar
                        st.markdown(f"""
                            <div style="margin-bottom:10px;">
                                <span style="color:{color}; font-weight:bold;">{sentiment.capitalize()}</span>
                                <div style="background-color:#e0e0e0; border-radius:10px; height:20px; width:100%;">
                                    <div style="background-color:{color}; width:{int(prob*100)}%; height:100%; border-radius:10px; text-align:center; color:white; line-height:20px;">
                                        {int(prob*100)}%
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Text preprocessing details
                with st.expander("Text Preprocessing Details"):
                    processed_text = preprocess_text(user_text)
                    st.markdown("**Original text:**")
                    st.write(user_text)
                    st.markdown("**Processed text:**")
                    st.write(processed_text)
    
    # Tab 2: CSV Upload
    with tab2:
        st.header("Batch Analysis with CSV Upload")
        
        st.info("Upload a CSV file with a column named 'review', 'text', or 'comment' containing the text to analyze.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            with st.spinner("Processing CSV file..."):
                result_df = process_csv(uploaded_file, model, vectorizer, sentiment_map)
                
                if result_df is not None:
                    st.success(f"Successfully processed {len(result_df)} reviews!")
                    
                    # Display results
                    st.subheader("Results Preview")
                    st.dataframe(result_df.head(10))
                    
                    # Summary statistics
                    st.subheader("Summary")
                    sentiment_counts = result_df['predicted_sentiment'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        positive_count = sentiment_counts.get('positive', 0)
                        st.markdown(f"""
                        <div style="background-color:#e6f7e9; padding:20px; border-radius:10px; text-align:center;">
                            <h3 style="color:green;">Positive</h3>
                            <p style="font-size:24px; color:green;">{positive_count}</p>
                            <p>({(positive_count/len(result_df)*100):.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        negative_count = sentiment_counts.get('negative', 0)
                        st.markdown(f"""
                        <div style="background-color:#feeeee; padding:20px; border-radius:10px; text-align:center;">
                            <h3 style="color:red;">Negative</h3>
                            <p style="font-size:24px; color:red;">{negative_count}</p>
                            <p>({(negative_count/len(result_df)*100):.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Data visualization
                    st.subheader("Visualization")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = {'positive': 'green', 'negative': 'red'}
                    result_df['predicted_sentiment'].value_counts().plot(kind='bar', color=[colors[x] for x in result_df['predicted_sentiment'].value_counts().index])
                    plt.title('Sentiment Distribution')
                    plt.xlabel('Sentiment')
                    plt.ylabel('Count')
                    st.pyplot(fig)
                    
                    # Download results
                    csv = result_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )
    
    # Tab 3: Model Performance
    with tab3:
        st.header("Model Performance")
    
        col1, col2 = st.columns(2)
        
    # Tab 4: Data Exploration
    with tab4:
        st.header("Data Exploration")
        
        st.subheader("Word Clouds by Sentiment")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Positive Sentiment Words")
            pos_wordcloud = generate_wordcloud(df, 'positive')
            if pos_wordcloud:
                st.image(pos_wordcloud)
        
        with col2:
            st.markdown("### Negative Sentiment Words")
            neg_wordcloud = generate_wordcloud(df, 'negative')
            if neg_wordcloud:
                st.image(neg_wordcloud)
    
        with col1:
            st.markdown(f"""
            <div style="background-color:white; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                <h3>Model Accuracy</h3>
                <p style="font-size:36px; color:#1e3a8a; font-weight:bold;">{accuracy:.2f}</p>
                <p>Based on test set evaluation</p>
            </div>
            """, unsafe_allow_html=True)
    
        with col2:
            st.markdown("""
            <div style="background-color:white; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                <h3>Model Details</h3>
                <p><strong>Algorithm:</strong> Logistic Regression</p>
                <p><strong>Vectorization:</strong> TF-IDF</p>
                <p><strong>Classes:</strong> Positive, Negative</p>
                <p><strong>Class Weighting:</strong> Balanced</p>
            </div>
            """, unsafe_allow_html=True)
    
        st.subheader("Top Words by Sentiment")
    
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
    
        if isinstance(model, LogisticRegression):
            # For binary classification: coefs[0] corresponds to positive class
            coefs = model.coef_[0]
    
            # Get top 10 positive features (largest coefficients)
            top_pos_indices = coefs.argsort()[-10:][::-1]
            top_pos_features = [(feature_names[i], coefs[i]) for i in top_pos_indices]
    
            # Get top 10 negative features (smallest coefficients)
            top_neg_indices = coefs.argsort()[:10]
            top_neg_features = [(feature_names[i], coefs[i]) for i in top_neg_indices]
    
            # --- Plot Positive Features ---
            st.subheader("Top Words for 'Positive' Sentiment")
            fig_pos, ax_pos = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(top_pos_features))
            ax_pos.barh(y_pos, [x[1] for x in top_pos_features], color='green')
            ax_pos.set_yticks(y_pos)
            ax_pos.set_yticklabels([x[0] for x in top_pos_features])
            ax_pos.invert_yaxis()
            ax_pos.set_xlabel("Coefficient Value")
            ax_pos.set_title("Top Positive Features")
            st.pyplot(fig_pos)
    
            # --- Plot Negative Features ---
            st.subheader("Top Words for 'Negative' Sentiment")
            fig_neg, ax_neg = plt.subplots(figsize=(10, 6))
            y_neg = np.arange(len(top_neg_features))
            ax_neg.barh(y_neg, [x[1] for x in top_neg_features], color='red')
            ax_neg.set_yticks(y_neg)
            ax_neg.set_yticklabels([x[0] for x in top_neg_features])
            ax_neg.invert_yaxis()
            ax_neg.set_xlabel("Coefficient Value")
            ax_neg.set_title("Top Negative Features")
            st.pyplot(fig_neg)

    
    # Tab 4: Data Exploration
    with tab4:
        st.header("Dataset Exploration")
        
        st.subheader("Dataset Preview")
        st.dataframe(df[['review_text', 'rating', 'sentiment']].head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            
            # Create sentiment distribution plot
            buf = plot_sentiment_distribution(df)
            st.image(buf, use_column_width=True)
        
        with col2:
            st.subheader("Rating Distribution")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(data=df, x='rating')
            plt.title('Rating Distribution')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            st.pyplot(fig)
        
        st.subheader("Word Clouds by Sentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Positive Sentiment")
            positive_wc = generate_wordcloud(df, 'positive')
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(positive_wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Negative Sentiment")
            negative_wc = generate_wordcloud(df, 'negative')
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(negative_wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("If you're seeing NLTK errors, please run the following in a Python terminal before running this app:")
    st.code("""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
    """)
