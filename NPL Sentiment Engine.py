'''
Project: Financial NLP Sentiment Engine
Description: Applies Natural Language Processing to a Kaggle financial news dataset 
to quantify market sentiment. Computes compound polarity scores using VADER 
and aggregates daily overall market sentiment by financial instrument.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# 1. Data Ingestion
def load_financial_news(file_path):
    """
    Loads a Kaggle financial news dataset. 
    Expected columns: 'date', 'ticker', 'headline'
    """
    print(f"Loading financial news dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Kaggle dataset not found. Please verify the file path.")
        return None
        
    # Standardize column names for processing
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Ensure necessary columns exist; adapt to Kaggle dataset structure
    if 'headline' not in df.columns and 'title' in df.columns:
        df.rename(columns={'title': 'headline'}, inplace=True)
    if 'ticker' not in df.columns and 'stock' in df.columns:
        df.rename(columns={'stock': 'ticker'}, inplace=True)
        
    df.dropna(subset=['headline', 'ticker'], inplace=True)
    return df

# 2. NLP Sentiment Scoring
def analyze_sentiment(df):
    """
    Applies VADER Sentiment Analysis to compute a compound score for each headline.
    Classifies the score into Bullish, Bearish, or Neutral categories.
    """
    print("Initializing NLP Sentiment Analyzer...")
    analyzer = SentimentIntensityAnalyzer()
    
    # Calculate compound sentiment score (-1 to 1)
    df['sentiment_score'] = df['headline'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
    
    # Categorize based on financial thresholds
    def categorize_sentiment(score):
        if score > 0.05: return 'Bullish'
        elif score < -0.05: return 'Bearish'
        else: return 'Neutral'
        
    df['market_view'] = df['sentiment_score'].apply(categorize_sentiment)
    return df

# 3. Data Aggregation & Reporting
def generate_sentiment_report(df, top_n=10):
    """
    Aggregates sentiment by Ticker to create a market summary report.
    Filters for the most mentioned tickers to ensure statistical relevance.
    """
    print("\n" + "=" * 60)
    print("AGGREGATED FINANCIAL SENTIMENT REPORT")
    print("=" * 60)
    
    # Filter for tickers with sufficient news volume
    top_tickers = df['ticker'].value_counts().head(top_n).index
    filtered_df = df[df['ticker'].isin(top_tickers)]
    
    summary = filtered_df.groupby('ticker').agg(
        Average_Score=('sentiment_score', 'mean'),
        News_Volume=('headline', 'count')
    ).reset_index()
    
    summary['Overall_Stance'] = summary['Average_Score'].apply(
        lambda x: 'Bullish' if x > 0.05 else ('Bearish' if x < -0.05 else 'Neutral')
    )
    
    summary = summary.sort_values(by='Average_Score', ascending=False)
    
    for _, row in summary.iterrows():
        print(f"Ticker: {row['ticker']:<6} | Volume: {row['News_Volume']:<4} | "
              f"Net Score: {row['Average_Score']:>6.3f} | Stance: {row['Overall_Stance']}")
        
    return summary

# 4. Visualization Engine
def plot_sentiment_distribution(df):
    """
    Generates a visual distribution of the sentiment classes across the dataset.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        x='market_view', 
        data=df, 
        order=['Bearish', 'Neutral', 'Bullish'], 
        palette=['#ff6b6b', '#ced4da', '#51cf66']
    )
    
    plt.title('Macro Distribution of Financial News Sentiment')
    plt.xlabel('Market View')
    plt.ylabel('Volume of Headlines')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', 
                    xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.show()

# 5. Main Execution Block
if __name__ == '__main__':
    KAGGLE_FILE_PATH = 'kaggle_financial_news.csv'
    
    news_df = load_financial_news(KAGGLE_FILE_PATH)
    
    if news_df is not None:
        analyzed_df = analyze_sentiment(news_df)
        
        # Display sample of processed data
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nSample Processed Headlines:")
        print(analyzed_df[['ticker', 'sentiment_score', 'market_view', 'headline']].head(5))
        
        generate_sentiment_report(analyzed_df)
        plot_sentiment_distribution(analyzed_df)
