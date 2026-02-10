"""
Visualization Module

Creates comprehensive visualizations for phishing evolution analysis:
- Time series plots
- Heatmaps
- Word clouds
- Comparison charts
- Interactive dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class PhishingVisualizer:
    """
    Create visualizations for phishing evolution analysis.
    """
    
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        self.color_palette = sns.color_palette("husl", 8)
        
    def plot_temporal_trend(self, df: pd.DataFrame,
                          feature: str,
                          temporal_column: str = 'year',
                          title: Optional[str] = None,
                          ylabel: Optional[str] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Plot temporal trend for a single feature.
        
        Args:
            df: DataFrame with temporal data
            feature: Feature to plot
            temporal_column: Temporal column name
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save figure
        """
        # Aggregate by temporal period
        temporal_stats = df.groupby(temporal_column)[feature].agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean with confidence interval
        ax.plot(temporal_stats[temporal_column], temporal_stats['mean'],
               marker='o', linewidth=2, markersize=8, label='Mean', color=self.color_palette[0])
        
        # Add confidence interval
        ax.fill_between(temporal_stats[temporal_column],
                       temporal_stats['mean'] - temporal_stats['std'],
                       temporal_stats['mean'] + temporal_stats['std'],
                       alpha=0.3, color=self.color_palette[0])
        
        # Formatting
        ax.set_xlabel(temporal_column.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel or feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(title or f'{feature.replace("_", " ").title()} Evolution Over Time',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_multiple_trends(self, df: pd.DataFrame,
                           features: List[str],
                           temporal_column: str = 'year',
                           title: str = 'Feature Evolution Over Time',
                           normalize: bool = True,
                           save_path: Optional[str] = None) -> None:
        """
        Plot multiple features on the same chart.
        
        Args:
            df: DataFrame with temporal data
            features: List of features to plot
            temporal_column: Temporal column name
            title: Plot title
            normalize: Whether to normalize features to 0-1 scale
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for i, feature in enumerate(features):
            if feature not in df.columns:
                continue
            
            temporal_stats = df.groupby(temporal_column)[feature].mean().reset_index()
            
            values = temporal_stats[feature].values
            
            # Normalize if requested
            if normalize and values.max() != values.min():
                values = (values - values.min()) / (values.max() - values.min())
            
            ax.plot(temporal_stats[temporal_column], values,
                   marker='o', linewidth=2, markersize=6,
                   label=feature.replace('_', ' ').title(),
                   color=self.color_palette[i % len(self.color_palette)])
        
        ax.set_xlabel(temporal_column.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Value' if normalize else 'Value',
                     fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_heatmap(self, df: pd.DataFrame,
                    features: List[str],
                    temporal_column: str = 'year',
                    title: str = 'Feature Evolution Heatmap',
                    save_path: Optional[str] = None) -> None:
        """
        Create heatmap showing feature evolution over time.
        
        Args:
            df: DataFrame with temporal data
            features: List of features to include
            temporal_column: Temporal column name
            title: Plot title
            save_path: Path to save figure
        """
        # Create pivot table
        heatmap_data = []
        
        for feature in features:
            if feature in df.columns:
                temporal_stats = df.groupby(temporal_column)[feature].mean()
                
                # Normalize to 0-1 scale
                if temporal_stats.max() != temporal_stats.min():
                    normalized = (temporal_stats - temporal_stats.min()) / (
                        temporal_stats.max() - temporal_stats.min()
                    )
                else:
                    normalized = temporal_stats
                
                heatmap_data.append(normalized)
        
        heatmap_df = pd.DataFrame(heatmap_data,
                                 index=[f.replace('_', ' ').title() for f in features])
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(features) * 0.5)))
        
        sns.heatmap(heatmap_df, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Normalized Value'},
                   linewidths=0.5, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(temporal_column.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame,
                              title: str = 'Feature Correlation Matrix',
                              save_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_comparison_bars(self, comparison_data: Dict,
                           title: str = 'Period Comparison',
                           save_path: Optional[str] = None) -> None:
        """
        Create bar chart comparing two time periods.
        
        Args:
            comparison_data: Dictionary with comparison statistics
            title: Plot title
            save_path: Path to save figure
        """
        features = [d['feature'] for d in comparison_data]
        period1_means = [d['period1_mean'] for d in comparison_data]
        period2_means = [d['period2_mean'] for d in comparison_data]
        
        x = np.arange(len(features))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width/2, period1_means, width,
                      label=f"Period 1", color=self.color_palette[0])
        bars2 = ax.bar(x + width/2, period2_means, width,
                      label=f"Period 2", color=self.color_palette[1])
        
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('_', ' ').title() for f in features],
                          rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def plot_distribution_evolution(self, df: pd.DataFrame,
                                   feature: str,
                                   temporal_column: str = 'year',
                                   title: Optional[str] = None,
                                   save_path: Optional[str] = None) -> None:
        """
        Plot distribution evolution using violin plots.
        
        Args:
            df: DataFrame with data
            feature: Feature to plot
            temporal_column: Temporal column name
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        sns.violinplot(data=df, x=temporal_column, y=feature, ax=ax,
                      palette='Set2', inner='quartile')
        
        ax.set_xlabel(temporal_column.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(title or f'{feature.replace("_", " ").title()} Distribution Evolution',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def create_dashboard(self, df: pd.DataFrame,
                        key_features: List[str],
                        temporal_column: str = 'year',
                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive dashboard with multiple subplots.
        
        Args:
            df: DataFrame with all data
            key_features: List of key features to display
            temporal_column: Temporal column name
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall trends (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        for i, feature in enumerate(key_features[:4]):
            if feature in df.columns:
                temporal_stats = df.groupby(temporal_column)[feature].mean()
                values = temporal_stats.values
                values_norm = (values - values.min()) / (values.max() - values.min()) if values.max() != values.min() else values
                ax1.plot(temporal_stats.index, values_norm,
                        marker='o', linewidth=2, label=feature.replace('_', ' ').title(),
                        color=self.color_palette[i])
        ax1.set_title('Key Phishing Evolution Trends', fontsize=14, fontweight='bold')
        ax1.set_xlabel(temporal_column.capitalize(), fontsize=11)
        ax1.set_ylabel('Normalized Value', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2-4. Individual feature trends (middle row)
        for i, feature in enumerate(key_features[:3]):
            ax = fig.add_subplot(gs[1, i])
            if feature in df.columns:
                temporal_stats = df.groupby(temporal_column)[feature].mean()
                ax.plot(temporal_stats.index, temporal_stats.values,
                       marker='o', linewidth=2, color=self.color_palette[i])
                ax.set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                ax.set_xlabel(temporal_column.capitalize(), fontsize=9)
                ax.grid(True, alpha=0.3)
        
        # 5-7. Distribution plots (bottom row)
        for i, feature in enumerate(key_features[3:6]):
            ax = fig.add_subplot(gs[2, i])
            if feature in df.columns:
                df.boxplot(column=feature, by=temporal_column, ax=ax)
                ax.set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                ax.set_xlabel(temporal_column.capitalize(), fontsize=9)
                plt.sca(ax)
                plt.xticks(rotation=45)
        
        fig.suptitle('Phishing Email Evolution Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()


def create_all_visualizations(df: pd.DataFrame,
                             temporal_column: str = 'year',
                             output_dir: str = 'visualizations') -> None:
    """
    Create all standard visualizations for the analysis.
    
    Args:
        df: Processed DataFrame
        temporal_column: Temporal column to use
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = PhishingVisualizer(output_dir)
    
    print("Creating visualizations...")
    
    # Define feature groups
    readability_features = ['flesch_reading_ease', 'gunning_fog', 'automated_readability_index']
    language_features = ['lexical_diversity', 'avg_word_length', 'avg_sentence_length']
    sentiment_features = ['sentiment_polarity', 'sentiment_subjectivity']
    grammar_features = ['spelling_error_rate', 'capitalization_errors', 'punctuation_density']
    psychological_features = ['urgency_density', 'fear_density', 'authority_density', 'reward_density']
    url_features = ['is_https', 'subdomain_count', 'url_length', 'is_typosquatting']
    
    # 1. Readability evolution
    visualizer.plot_multiple_trends(
        df, readability_features, temporal_column,
        title='Readability Score Evolution',
        save_path=f'{output_dir}/readability_evolution.png'
    )
    
    # 2. Language complexity
    visualizer.plot_multiple_trends(
        df, language_features, temporal_column,
        title='Language Complexity Evolution',
        save_path=f'{output_dir}/language_complexity.png'
    )
    
    # 3. Psychological triggers
    visualizer.plot_multiple_trends(
        df, psychological_features, temporal_column,
        title='Psychological Trigger Evolution',
        save_path=f'{output_dir}/psychological_triggers.png'
    )
    
    # 4. URL sophistication
    if all(f in df.columns for f in url_features):
        visualizer.plot_multiple_trends(
            df, url_features, temporal_column,
            title='URL Sophistication Evolution',
            save_path=f'{output_dir}/url_sophistication.png'
        )
    
    # 5. Comprehensive dashboard
    key_features = [
        'flesch_reading_ease', 'lexical_diversity', 'urgency_density',
        'is_https', 'subdomain_count', 'spelling_error_rate'
    ]
    visualizer.create_dashboard(
        df, key_features, temporal_column,
        save_path=f'{output_dir}/comprehensive_dashboard.png'
    )
    
    print(f"All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    print("Visualization Module")
    print("This module creates comprehensive visualizations for phishing evolution analysis")