"""
NLP Features Extraction Module

Extracts linguistic and stylistic features from phishing emails to track evolution:
- Readability scores
- Lexical diversity
- Sentiment analysis
- Grammar quality
- Professional language indicators
- Psychological trigger detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from textblob import TextBlob
    import textstat
except ImportError:
    print("Warning: Some NLP libraries not installed. Install with: pip install nltk textblob textstat")


class NLPFeatureExtractor:
    """
    Extract linguistic features from phishing emails for temporal analysis.
    """
    
    def __init__(self):
        """Initialize the NLP feature extractor."""
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        
        # Define psychological trigger keywords
        self.urgency_keywords = [
            'urgent', 'immediate', 'immediately', 'now', 'asap', 'hurry',
            'quick', 'quickly', 'expire', 'expiring', 'deadline', 'limited',
            'act now', 'don\'t wait', 'time-sensitive', 'last chance'
        ]
        
        self.authority_keywords = [
            'bank', 'paypal', 'amazon', 'irs', 'government', 'security',
            'administrator', 'manager', 'official', 'department', 'team',
            'support', 'service', 'microsoft', 'apple', 'google'
        ]
        
        self.fear_keywords = [
            'suspended', 'locked', 'blocked', 'unauthorized', 'suspicious',
            'fraud', 'compromised', 'violated', 'breach', 'hacked',
            'illegal', 'lawsuit', 'penalty', 'criminal', 'investigation'
        ]
        
        self.reward_keywords = [
            'winner', 'won', 'prize', 'reward', 'gift', 'free',
            'congratulations', 'selected', 'lucky', 'bonus', 'claim',
            'refund', 'compensation', 'discount', 'offer'
        ]
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
    
    def extract_all_features(self, df: pd.DataFrame, 
                           text_column: str = 'body') -> pd.DataFrame:
        """
        Extract all NLP features from text data.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with added NLP features
        """
        print("Extracting NLP features...")
        
        if text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in DataFrame")
        
        # Readability features
        print("  - Calculating readability scores...")
        df = self._add_readability_features(df, text_column)
        
        # Lexical features
        print("  - Extracting lexical features...")
        df = self._add_lexical_features(df, text_column)
        
        # Sentiment analysis
        print("  - Analyzing sentiment...")
        df = self._add_sentiment_features(df, text_column)
        
        # Grammar quality
        print("  - Assessing grammar quality...")
        df = self._add_grammar_features(df, text_column)
        
        # Psychological triggers
        print("  - Detecting psychological triggers...")
        df = self._add_psychological_features(df, text_column)
        
        # Professional language indicators
        print("  - Measuring professionalism...")
        df = self._add_professionalism_features(df, text_column)
        
        print("Feature extraction complete!")
        return df
    
    def _add_readability_features(self, df: pd.DataFrame, 
                                 text_column: str) -> pd.DataFrame:
        """Add readability metrics."""
        
        df['flesch_reading_ease'] = df[text_column].apply(
            lambda x: textstat.flesch_reading_ease(str(x)) if pd.notna(x) else 0
        )
        
        df['flesch_kincaid_grade'] = df[text_column].apply(
            lambda x: textstat.flesch_kincaid_grade(str(x)) if pd.notna(x) else 0
        )
        
        df['gunning_fog'] = df[text_column].apply(
            lambda x: textstat.gunning_fog(str(x)) if pd.notna(x) else 0
        )
        
        df['automated_readability_index'] = df[text_column].apply(
            lambda x: textstat.automated_readability_index(str(x)) if pd.notna(x) else 0
        )
        
        return df
    
    def _add_lexical_features(self, df: pd.DataFrame, 
                            text_column: str) -> pd.DataFrame:
        """Add lexical diversity and complexity features."""
        
        def calculate_lexical_diversity(text):
            """Type-Token Ratio (TTR)."""
            if pd.isna(text) or text == '':
                return 0
            tokens = word_tokenize(str(text).lower())
            if len(tokens) == 0:
                return 0
            return len(set(tokens)) / len(tokens)
        
        def average_word_length(text):
            """Average word length."""
            if pd.isna(text) or text == '':
                return 0
            words = word_tokenize(str(text))
            if len(words) == 0:
                return 0
            return np.mean([len(word) for word in words])
        
        def average_sentence_length(text):
            """Average sentence length."""
            if pd.isna(text) or text == '':
                return 0
            sentences = sent_tokenize(str(text))
            if len(sentences) == 0:
                return 0
            words = word_tokenize(str(text))
            return len(words) / len(sentences)
        
        df['lexical_diversity'] = df[text_column].apply(calculate_lexical_diversity)
        df['avg_word_length'] = df[text_column].apply(average_word_length)
        df['avg_sentence_length'] = df[text_column].apply(average_sentence_length)
        df['word_count'] = df[text_column].apply(
            lambda x: len(word_tokenize(str(x))) if pd.notna(x) else 0
        )
        df['sentence_count'] = df[text_column].apply(
            lambda x: len(sent_tokenize(str(x))) if pd.notna(x) else 0
        )
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, 
                               text_column: str) -> pd.DataFrame:
        """Add sentiment analysis features."""
        
        def get_sentiment(text):
            """Get sentiment polarity and subjectivity."""
            if pd.isna(text) or text == '':
                return 0, 0
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        
        sentiments = df[text_column].apply(get_sentiment)
        df['sentiment_polarity'] = sentiments.apply(lambda x: x[0])
        df['sentiment_subjectivity'] = sentiments.apply(lambda x: x[1])
        
        return df
    
    def _add_grammar_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Add lightweight grammar quality indicators (FAST)."""
        def punctuation_density(text):
            if pd.isna(text) or text == '':
                return 0
            text = str(text)
            return sum(1 for c in text if c in '.,!?;:') / max(len(text), 1)

        def capitalization_ratio(text):
            if pd.isna(text) or text == '':
                return 0
            text = str(text)
            return sum(1 for c in text if c.isupper()) / max(len(text), 1)

        def repeated_exclamation(text):
            if pd.isna(text) or text == '':
                return 0
            return 1 if '!!' in str(text) or '???' in str(text) else 0

        df['punctuation_density'] = df[text_column].apply(punctuation_density)
        df['capitalization_ratio'] = df[text_column].apply(capitalization_ratio)
        df['repeated_exclamation'] = df[text_column].apply(repeated_exclamation)

        return df

    
    def _add_psychological_features(self, df: pd.DataFrame, 
                                   text_column: str) -> pd.DataFrame:
        """Add psychological trigger detection features."""
        
        def count_keywords(text, keywords):
            """Count occurrences of keywords in text."""
            if pd.isna(text) or text == '':
                return 0
            text_lower = str(text).lower()
            return sum(1 for keyword in keywords if keyword in text_lower)
        
        df['urgency_score'] = df[text_column].apply(
            lambda x: count_keywords(x, self.urgency_keywords)
        )
        
        df['authority_score'] = df[text_column].apply(
            lambda x: count_keywords(x, self.authority_keywords)
        )
        
        df['fear_score'] = df[text_column].apply(
            lambda x: count_keywords(x, self.fear_keywords)
        )
        
        df['reward_score'] = df[text_column].apply(
            lambda x: count_keywords(x, self.reward_keywords)
        )
        
        # Normalize by word count
        df['urgency_density'] = df['urgency_score'] / (df['word_count'] + 1)
        df['authority_density'] = df['authority_score'] / (df['word_count'] + 1)
        df['fear_density'] = df['fear_score'] / (df['word_count'] + 1)
        df['reward_density'] = df['reward_score'] / (df['word_count'] + 1)
        
        return df
    
    def _add_professionalism_features(self, df: pd.DataFrame, 
                                     text_column: str) -> pd.DataFrame:
        """Add professional language indicators."""
        
        # Professional salutations
        professional_greetings = [
            'dear', 'hello', 'greetings', 'good morning', 'good afternoon'
        ]
        
        # Professional closings
        professional_closings = [
            'sincerely', 'regards', 'best regards', 'cordially', 'respectfully'
        ]
        
        def has_professional_greeting(text):
            """Check for professional greeting."""
            if pd.isna(text) or text == '':
                return 0
            text_lower = str(text).lower()
            return 1 if any(greeting in text_lower for greeting in professional_greetings) else 0
        
        def has_professional_closing(text):
            """Check for professional closing."""
            if pd.isna(text) or text == '':
                return 0
            text_lower = str(text).lower()
            return 1 if any(closing in text_lower for closing in professional_closings) else 0
        
        def formality_score(text):
            """Estimate formality based on vocabulary."""
            if pd.isna(text) or text == '':
                return 0
            
            # Formal indicators
            formal_words = [
                'furthermore', 'therefore', 'however', 'consequently',
                'additionally', 'moreover', 'nevertheless', 'accordingly'
            ]
            
            text_lower = str(text).lower()
            formal_count = sum(1 for word in formal_words if word in text_lower)
            
            # Informal indicators (negative score)
            informal_words = ['gonna', 'wanna', 'yeah', 'yep', 'nope', 'ok', 'lol']
            informal_count = sum(1 for word in informal_words if word in text_lower)
            
            return formal_count - informal_count
        
        df['has_greeting'] = df[text_column].apply(has_professional_greeting)
        df['has_closing'] = df[text_column].apply(has_professional_closing)
        df['formality_score'] = df[text_column].apply(formality_score)
        
        return df
    
    def extract_top_keywords(self, df: pd.DataFrame, 
                           text_column: str = 'body',
                           n_keywords: int = 50) -> Dict[str, int]:
        """
        Extract top keywords from all texts.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text
            n_keywords: Number of top keywords to extract
            
        Returns:
            Dictionary of top keywords and their counts
        """
        print(f"Extracting top {n_keywords} keywords...")
        
        all_words = []
        
        for text in df[text_column]:
            if pd.notna(text):
                tokens = word_tokenize(str(text).lower())
                # Filter out stopwords and non-alphabetic tokens
                words = [word for word in tokens 
                        if word.isalpha() and word not in self.stop_words and len(word) > 3]
                all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_keywords = dict(word_freq.most_common(n_keywords))
        
        return top_keywords
    
    def analyze_keyword_evolution(self, df: pd.DataFrame,
                                 text_column: str = 'body',
                                 temporal_column: str = 'year',
                                 n_keywords: int = 20) -> pd.DataFrame:
        """
        Analyze how top keywords evolve over time.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text
            temporal_column: Temporal grouping column
            n_keywords: Number of keywords to track
            
        Returns:
            DataFrame with keyword frequencies per time period
        """
        print(f"Analyzing keyword evolution over {temporal_column}...")
        
        # Get overall top keywords
        top_keywords = self.extract_top_keywords(df, text_column, n_keywords)
        keywords_to_track = list(top_keywords.keys())
        
        # Calculate frequency per time period
        evolution_data = []
        
        for period in sorted(df[temporal_column].unique()):
            period_df = df[df[temporal_column] == period]
            period_text = ' '.join(period_df[text_column].astype(str))
            period_tokens = word_tokenize(period_text.lower())
            
            period_freq = Counter(period_tokens)
            
            row = {'period': period}
            for keyword in keywords_to_track:
                row[keyword] = period_freq.get(keyword, 0)
            
            evolution_data.append(row)
        
        evolution_df = pd.DataFrame(evolution_data)
        
        return evolution_df


if __name__ == "__main__":
    # Example usage
    print("NLP Feature Extractor Module")
    print("This module extracts linguistic features for phishing evolution analysis")