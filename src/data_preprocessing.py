"""
Data Preprocessing Module for Phishing Evolution Analyzer

This module handles data loading, cleaning, and preprocessing for phishing email datasets.
It prepares data for temporal analysis by extracting dates, cleaning text, and standardizing formats.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class PhishingDataPreprocessor:
    """
    Preprocesses phishing email and URL datasets for temporal analysis.
    """
    
    def __init__(self, email_data_path: Optional[str] = None, 
                 url_data_path: Optional[str] = None):
        """
        Initialize the preprocessor with dataset paths.
        
        Args:
            email_data_path: Path to email dataset CSV
            url_data_path: Path to URL dataset CSV
        """
        self.email_data_path = email_data_path
        self.url_data_path = url_data_path
        self.email_df = None
        self.url_df = None
        
    def load_email_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and perform initial cleaning of email dataset.
        
        Args:
            data_path: Path to CSV file (overrides initialization path)
            
        Returns:
            Preprocessed email DataFrame
        """
        path = data_path or self.email_data_path
        
        if path is None:
            raise ValueError("No data path provided")
        
        print(f"Loading email data from {path}...")
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle different dataset formats
        self._standardize_email_columns(df)
        
        self.email_df = df
        print(f"Loaded {len(df)} email records")
        return df
    
    def _standardize_email_columns(self, df: pd.DataFrame) -> None:
        """
        Standardize column names across different dataset formats.
        """
        # Common column mappings
        column_mappings = {
            'subject': ['subject', 'subject_line', 'email_subject'],
            'body': ['body', 'email_body', 'text', 'content', 'message'],
            'label': ['label', 'class', 'type', 'is_phishing', 'phishing'],
            'date': ['date', 'timestamp', 'sent_date', 'email_date', 'year'],
            'from': ['from', 'sender', 'from_address', 'email_from'],
            'url': ['url', 'urls', 'link', 'links']
        }
        
        for standard_name, variations in column_mappings.items():
            for var in variations:
                if var in df.columns and standard_name not in df.columns:
                    df.rename(columns={var: standard_name}, inplace=True)
                    break
    
    def extract_temporal_features(self, df: pd.DataFrame, 
                                  date_column: str = 'date') -> pd.DataFrame:
        """
        Extract temporal features from dates for time-series analysis.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            
        Returns:
            DataFrame with added temporal features
        """
        print("Extracting temporal features...")
        
        if date_column not in df.columns:
            print(f"Warning: {date_column} column not found. Attempting to infer dates...")
            df = self._infer_dates(df)
        
        # Convert to datetime
        df['date_parsed'] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Extract temporal components
        df['year'] = df['date_parsed'].dt.year
        df['month'] = df['date_parsed'].dt.month
        df['quarter'] = df['date_parsed'].dt.quarter
        df['day_of_week'] = df['date_parsed'].dt.dayofweek
        df['week_of_year'] = df['date_parsed'].dt.isocalendar().week
        
        # Remove rows with invalid dates
        valid_dates = df['year'].notna()
        print(f"Removed {(~valid_dates).sum()} records with invalid dates")
        df = df[valid_dates].copy()
        
        return df
    
    def _infer_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to infer dates from email content or other fields.
        """
        # Try to extract year from text content
        if 'body' in df.columns:
            df['date'] = df['body'].apply(self._extract_year_from_text)
        elif 'subject' in df.columns:
            df['date'] = df['subject'].apply(self._extract_year_from_text)
        else:
            # If no date info, create synthetic dates for demonstration
            df['date'] = pd.date_range(start='2015-01-01', 
                                       periods=len(df), 
                                       freq='D')
        return df
    
    def _extract_year_from_text(self, text: str) -> Optional[str]:
        """
        Extract year from text content.
        """
        if pd.isna(text):
            return None
        
        # Look for 4-digit years (2010-2024)
        year_pattern = r'\b(20[1-2][0-9])\b'
        match = re.search(year_pattern, str(text))
        
        if match:
            return f"{match.group(1)}-01-01"
        return None
    
    def clean_text(self, df: pd.DataFrame, 
                  text_columns: List[str] = ['subject', 'body']) -> pd.DataFrame:
        """
        Clean and normalize text columns.
        
        Args:
            df: Input DataFrame
            text_columns: List of text columns to clean
            
        Returns:
            DataFrame with cleaned text
        """
        print("Cleaning text data...")
        
        for col in text_columns:
            if col in df.columns:
                # Convert to string
                df[col] = df[col].astype(str)
                
                # Remove null values
                df[col] = df[col].replace('nan', '')
                
                # Basic cleaning (preserve original for some analyses)
                df[f'{col}_clean'] = df[col].apply(self._clean_text_content)
        
        return df
    
    def _clean_text_content(self, text: str) -> str:
        """
        Clean individual text content.
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (but keep basic punctuation)
        # text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        
        return text.strip()
    
    def extract_urls(self, df: pd.DataFrame, 
                    text_column: str = 'body') -> pd.DataFrame:
        """
        Extract URLs from email content.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text with URLs
            
        Returns:
            DataFrame with extracted URLs
        """
        print("Extracting URLs from emails...")
        
        if text_column not in df.columns:
            print(f"Warning: {text_column} not found")
            df['extracted_urls'] = None
            return df
        
        # URL regex pattern
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        df['extracted_urls'] = df[text_column].apply(
            lambda x: re.findall(url_pattern, str(x)) if pd.notna(x) else []
        )
        
        df['url_count'] = df['extracted_urls'].apply(len)
        
        return df
    
    def standardize_labels(self, df: pd.DataFrame, 
                          label_column: str = 'label') -> pd.DataFrame:
        """
        Standardize phishing labels to binary format.
        
        Args:
            df: Input DataFrame
            label_column: Name of label column
            
        Returns:
            DataFrame with standardized labels
        """
        print("Standardizing labels...")
        
        if label_column not in df.columns:
            print(f"Warning: {label_column} not found. Creating default labels.")
            df['is_phishing'] = 1  # Assume phishing dataset
            return df
        
        # Convert various label formats to binary
        df['is_phishing'] = df[label_column].apply(self._convert_to_binary_label)
        
        phishing_count = df['is_phishing'].sum()
        print(f"Dataset contains {phishing_count} phishing emails, "
              f"{len(df) - phishing_count} legitimate emails")
        
        return df
    
    def _convert_to_binary_label(self, label) -> int:
        """
        Convert various label formats to binary (1=phishing, 0=legitimate).
        """
        if pd.isna(label):
            return 1  # Default to phishing if unknown
        
        label_str = str(label).lower().strip()
        
        phishing_indicators = ['phishing', 'spam', 'phish', '1', 'true', 'yes']
        
        return 1 if any(ind in label_str for ind in phishing_indicators) else 0
    
    def create_temporal_splits(self, df: pd.DataFrame, 
                              split_by: str = 'year') -> dict:
        """
        Split data into temporal periods for evolution analysis.
        
        Args:
            df: Input DataFrame
            split_by: Temporal unit ('year', 'quarter', 'month')
            
        Returns:
            Dictionary of DataFrames split by temporal period
        """
        print(f"Creating temporal splits by {split_by}...")
        
        if split_by not in df.columns:
            raise ValueError(f"{split_by} column not found. Run extract_temporal_features first.")
        
        temporal_splits = {}
        
        for period in sorted(df[split_by].unique()):
            temporal_splits[period] = df[df[split_by] == period].copy()
            print(f"  {split_by} {period}: {len(temporal_splits[period])} records")
        
        return temporal_splits
    
    def process_complete_pipeline(self, 
                                 email_path: Optional[str] = None,
                                 output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.
        
        Args:
            email_path: Path to email dataset
            output_path: Path to save processed data
            
        Returns:
            Fully processed DataFrame
        """
        print("="*60)
        print("STARTING COMPLETE PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        df = self.load_email_data(email_path)
        
        # Step 2: Extract temporal features
        df = self.extract_temporal_features(df)
        
        # Step 3: Clean text
        df = self.clean_text(df)
        
        # Step 4: Extract URLs
        df = self.extract_urls(df)
        
        # Step 5: Standardize labels
        df = self.standardize_labels(df)
        
        # Save processed data
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nProcessed data saved to {output_path}")
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['year'].min()} - {df['year'].max()}")
        print(f"Columns: {list(df.columns)}")
        
        return df


def create_sample_dataset(output_path: str, num_samples: int = 1000) -> pd.DataFrame:
    """
    Create a sample phishing dataset for demonstration purposes.
    
    Args:
        output_path: Path to save the sample dataset
        num_samples: Number of sample records to generate
        
    Returns:
        Sample DataFrame
    """
    print(f"Creating sample dataset with {num_samples} records...")
    
    np.random.seed(42)
    
    # Generate dates from 2015 to 2024
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', periods=num_samples)
    
    # Sample subjects (phishing)
    subjects = [
        "Urgent: Verify your account",
        "Your account has been compromised",
        "Confirm your identity immediately",
        "Important security alert",
        "Action required: Update payment information",
        "Your package is waiting",
        "Congratulations! You've won",
        "Reset your password now",
        "Unusual activity detected",
        "Click here to claim your prize"
    ]
    
    # Sample bodies
    bodies = [
        "Dear valued customer, we have detected suspicious activity on your account. Please verify your identity by clicking the link below: http://secure-bank-verify.com/login",
        "Your account will be suspended unless you update your payment information immediately. Click here: https://paypal-secure.verify-account.com",
        "Congratulations! You have been selected as a winner. Claim your prize now: http://prize-winner.net/claim",
        "We need to confirm your identity. Please provide your account details: https://account-security.com/verify",
        "Your package is ready for delivery. Track your shipment: http://tracking-express.com/package",
    ]
    
    data = {
        'date': dates,
        'subject': np.random.choice(subjects, num_samples),
        'body': np.random.choice(bodies, num_samples),
        'label': 'phishing',
        'from': [f"noreply{i}@suspicious-domain.com" for i in range(num_samples)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Example usage
    preprocessor = PhishingDataPreprocessor()
    
    # Create sample data for demonstration
    sample_df = create_sample_dataset('data/raw/sample_phishing_emails.csv')
    
    # Process the data
    processed_df = preprocessor.process_complete_pipeline(
        email_path='data/raw/sample_phishing_emails.csv',
        output_path='data/processed/phishing_emails_processed.csv'
    )
    
    print("\nSample of processed data:")
    print(processed_df.head())