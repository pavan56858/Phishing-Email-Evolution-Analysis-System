"""
Main Execution Script - Phishing Evolution Analyzer

This script orchestrates the complete analysis pipeline:
1. Data preprocessing
2. Feature extraction (NLP + URL)
3. Temporal analysis
4. Visualization generation
5. Report creation
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from data_preprocessing import PhishingDataPreprocessor, create_sample_dataset
from nlp_features import NLPFeatureExtractor
from url_analyzer import URLAnalyzer
from temporal_analyzer import TemporalAnalyzer
from visualization import create_all_visualizations, PhishingVisualizer

def generate_trend_insight(feature, start, end,
                           up_msg, down_msg, stable_msg,
                           threshold=0.05):
    """
    Auto-generate insight based on feature trend.
    """
    if pd.isna(start) or pd.isna(end):
        return None

    change_ratio = (end - start) / (abs(start) + 1e-6)

    if change_ratio > threshold:
        return f"• {up_msg}"
    elif change_ratio < -threshold:
        return f"• {down_msg}"
    else:
        return f"• {stable_msg}"


class PhishingEvolutionAnalyzer:
    """
    Main class orchestrating the complete phishing evolution analysis.
    """
    
    def __init__(self, data_path: str = None, use_sample_data: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to phishing email dataset
            use_sample_data: Whether to create and use sample data
        """
        self.data_path = data_path
        self.use_sample_data = use_sample_data
        self.df = None
        self.preprocessor = PhishingDataPreprocessor()
        self.nlp_extractor = NLPFeatureExtractor()
        self.url_analyzer = URLAnalyzer()
        self.temporal_analyzer = None
        
        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary project directories."""
        directories = [
            'data/raw',
            'data/processed',
            'data/features',
            'visualizations',
            'reports',
            'models'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_complete_analysis(self, temporal_column: str = 'year') -> pd.DataFrame:
        """
        Run the complete analysis pipeline.
        
        Args:
            temporal_column: Temporal column to use for analysis
            
        Returns:
            Processed DataFrame with all features
        """
        print("\n" + "="*70)
        print("PHISHING EVOLUTION ANALYZER - COMPLETE PIPELINE")
        print("="*70 + "\n")
        
        # Step 1: Load or create data
        print("STEP 1: DATA LOADING")
        print("-" * 70)
        if self.use_sample_data:
            print("Creating sample dataset...")
            create_sample_dataset('data/raw/sample_phishing_emails.csv', num_samples=2000)
            self.data_path = 'data/raw/sample_phishing_emails.csv'
        
        # Step 2: Preprocess data
        print("\nSTEP 2: DATA PREPROCESSING")
        print("-" * 70)
        self.df = self.preprocessor.process_complete_pipeline(
            email_path=self.data_path,
            output_path='data/processed/phishing_emails_processed.csv'
        )
        
        # Step 3: Extract NLP features
        print("\nSTEP 3: NLP FEATURE EXTRACTION")
        print("-" * 70)
        self.df = self.nlp_extractor.extract_all_features(self.df, text_column='body')
        
        # Step 4: Extract URL features
        print("\nSTEP 4: URL FEATURE EXTRACTION")
        print("-" * 70)
        self.df = self.url_analyzer.extract_url_features(self.df)
        
        # Save feature-enriched data
        self.df.to_csv('data/features/phishing_with_features.csv', index=False)
        print(f"Feature-enriched data saved: {len(self.df.columns)} columns")
        
        # Step 5: Temporal analysis
        print("\nSTEP 5: TEMPORAL ANALYSIS")
        print("-" * 70)
        self.temporal_analyzer = TemporalAnalyzer(self.df)
        
        # Analyze key features
        key_features = self._get_key_features()
        trends_df = self.temporal_analyzer.analyze_feature_trends(
            key_features, temporal_column
        )
        
        print("\nTop Trends Identified:")
        print(trends_df[['feature', 'trend_direction', 'trend_strength', 
                        'percent_change']].head(10).to_string())
        
        # Save trends
        trends_df.to_csv('reports/feature_trends.csv', index=False)
        
        # Step 6: Generate visualizations
        print("\nSTEP 6: VISUALIZATION GENERATION")
        print("-" * 70)
        create_all_visualizations(self.df, temporal_column, 'visualizations')
        
        # Step 7: Generate comprehensive report
        print("\nSTEP 7: REPORT GENERATION")
        print("-" * 70)
        report = self.generate_comprehensive_report(temporal_column)
        
        with open('reports/phishing_evolution_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("Report saved to: reports/phishing_evolution_report.txt")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to:")
        print("  - Processed data: data/features/phishing_with_features.csv")
        print("  - Trends: reports/feature_trends.csv")
        print("  - Visualizations: visualizations/")
        print("  - Report: reports/phishing_evolution_report.txt")
        
        return self.df
    
    def _get_key_features(self) -> list:
        """Get list of key features for analysis."""
        return [
            # Readability
            'flesch_reading_ease',
            'flesch_kincaid_grade',
            'gunning_fog',
            
            # Language
            'lexical_diversity',
            'avg_word_length',
            'avg_sentence_length',
            
            # Sentiment
            'sentiment_polarity',
            'sentiment_subjectivity',
            
            # Grammar
            
            
            # Psychological
            'urgency_density',
            'fear_density',
            'authority_density',
            'reward_density',
            
            # Professionalism
            'has_greeting',
            'has_closing',
            'formality_score',
            
            # URL
            'is_https',
            'subdomain_count',
            'url_length',
            'is_typosquatting',
            'misleading_keywords'
        ]
    
    def generate_comprehensive_report(self, temporal_column: str = 'year') -> str:

        report = []
        report.append("="*80)
        report.append("PHISHING EMAIL EVOLUTION ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n" + "="*80)

        # 1. Dataset summary
        report.append("\n1. DATASET SUMMARY")
        report.append("-"*80)
        report.append(f"Total emails analyzed: {len(self.df)}")
        report.append(f"Time period: {self.df[temporal_column].min()} - {self.df[temporal_column].max()}")
        report.append(f"Phishing emails: {self.df['is_phishing'].sum()}")
        report.append(f"Legitimate emails: {(~self.df['is_phishing'].astype(bool)).sum()}")

        # 2. Temporal trends
        report.append("\n2. KEY TEMPORAL TRENDS")
        report.append("-"*80)
        report.append(
            self.temporal_analyzer.generate_summary_report(
                self._get_key_features(), temporal_column
            )
        )

        # 3. URL evolution
        report.append("\n3. URL SOPHISTICATION EVOLUTION")
        report.append("-"*80)
        url_evolution = self.url_analyzer.analyze_url_evolution(self.df, temporal_column)

        report.append(
            f"HTTPS Adoption: {url_evolution['https_percentage'].iloc[0]:.1f}% → "
            f"{url_evolution['https_percentage'].iloc[-1]:.1f}%"
        )
        report.append(
            f"Average Subdomains: {url_evolution['avg_subdomain_count'].iloc[0]:.2f} → "
            f"{url_evolution['avg_subdomain_count'].iloc[-1]:.2f}"
        )
        report.append(
            f"Average URL Length: {url_evolution['avg_url_length'].iloc[0]:.0f} → "
            f"{url_evolution['avg_url_length'].iloc[-1]:.0f}"
        )

        # 4. Language evolution
        report.append("\n4. LANGUAGE SOPHISTICATION")
        report.append("-"*80)

        first_year = self.df[temporal_column].min()
        last_year = self.df[temporal_column].max()

        first = self.df[self.df[temporal_column] == first_year]
        last = self.df[self.df[temporal_column] == last_year]

        report.append(
            f"Readability (Flesch): {first['flesch_reading_ease'].mean():.1f} → "
            f"{last['flesch_reading_ease'].mean():.1f}"
        )
        report.append(
            f"Lexical Diversity: {first['lexical_diversity'].mean():.3f} → "
            f"{last['lexical_diversity'].mean():.3f}"
        )

        # 5. Psychological tactics
        report.append("\n5. PSYCHOLOGICAL TACTICS EVOLUTION")
        report.append("-"*80)
        report.append(
            f"Urgency Tactics: {first['urgency_density'].mean():.4f} → "
            f"{last['urgency_density'].mean():.4f}"
        )
        report.append(
            f"Fear Tactics: {first['fear_density'].mean():.4f} → "
            f"{last['fear_density'].mean():.4f}"
        )
        report.append(
            f"Authority Impersonation: {first['authority_density'].mean():.4f} → "
            f"{last['authority_density'].mean():.4f}"
        )

        # 6. AUTO-GENERATED KEY INSIGHTS
        report.append("\n6. KEY INSIGHTS")
        report.append("-"*80)

        insights = [
            generate_trend_insight(
                "lexical_diversity",
                first["lexical_diversity"].mean(),
                last["lexical_diversity"].mean(),
                "Phishing emails have become more linguistically professional over time",
                "Phishing language quality has degraded over time",
                "Linguistic sophistication of phishing emails remains stable"
            ),
            generate_trend_insight(
                "is_https",
                first["is_https"].mean(),
                last["is_https"].mean(),
                "Phishing URLs increasingly use HTTPS to appear legitimate",
                "Phishing URLs rely less on HTTPS infrastructure",
                "HTTPS usage in phishing URLs remains consistent"
            ),
            generate_trend_insight(
                "subdomain_count",
                first["subdomain_count"].mean(),
                last["subdomain_count"].mean(),
                "Phishing URLs have become structurally more complex",
                "Phishing URLs have become structurally simpler",
                "URL structural complexity remains unchanged"
            ),
            generate_trend_insight(
                "urgency_density",
                first["urgency_density"].mean(),
                last["urgency_density"].mean(),
                "Use of urgency-based psychological manipulation has increased",
                "Urgency-based manipulation tactics have decreased",
                "Psychological urgency tactics remain stable"
            )
        ]

        for i in insights:
            if i:
                report.append(i)

        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)

        return "\n".join(report)
    

    
    def analyze_specific_period(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Analyze a specific time period.
        
        Args:
            start_year: Start year
            end_year: End year
            
        Returns:
            Filtered DataFrame for the period
        """
        period_df = self.df[
            (self.df['year'] >= start_year) & (self.df['year'] <= end_year)
        ]
        
        print(f"\nAnalysis for {start_year}-{end_year}:")
        print(f"Total emails: {len(period_df)}")
        print(f"Average readability: {period_df['flesch_reading_ease'].mean():.2f}")
        print(f"HTTPS usage: {period_df['is_https'].mean()*100:.1f}%")
        
        return period_df


def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = PhishingEvolutionAnalyzer(use_sample_data=True)
    
    # Run complete analysis
    df = analyzer.run_complete_analysis()
    
    # Optional: Analyze specific periods
    print("\n" + "="*70)
    print("PERIOD-SPECIFIC ANALYSIS")
    print("="*70)
    analyzer.analyze_specific_period(2015, 2019)
    analyzer.analyze_specific_period(2020, 2024)
    
    return analyzer, df


if __name__ == "__main__":
    analyzer, df = main()
    print("\nAnalysis object available as 'analyzer'")
    print("Data available as 'df'")