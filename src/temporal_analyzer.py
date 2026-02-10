"""
Temporal Analysis Module

Analyzes temporal trends in phishing email characteristics:
- Year-over-year evolution
- Quarterly and monthly trends
- Statistical trend detection
- Correlation analysis between features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class TemporalAnalyzer:
    """
    Analyze temporal evolution of phishing email characteristics.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize temporal analyzer with preprocessed data.
        
        Args:
            df: Preprocessed DataFrame with temporal and feature columns
        """
        self.df = df
        self.temporal_column = 'year'
        self.trends = {}
        
    def analyze_feature_trends(self, features: List[str],
                              temporal_column: str = 'year') -> pd.DataFrame:
        """
        Analyze trends for multiple features over time.
        
        Args:
            features: List of feature column names to analyze
            temporal_column: Column to group by (year, quarter, month)
            
        Returns:
            DataFrame with trend statistics
        """
        print(f"Analyzing trends for {len(features)} features over {temporal_column}...")
        
        self.temporal_column = temporal_column
        trends_data = []
        
        for feature in features:
            if feature not in self.df.columns:
                print(f"Warning: Feature {feature} not found, skipping...")
                continue
            
            # Calculate temporal statistics
            temporal_stats = self.df.groupby(temporal_column)[feature].agg([
                'mean', 'median', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            # Calculate trend direction and strength
            trend_info = self._calculate_trend(temporal_stats, temporal_column, 'mean')
            
            trend_data = {
                'feature': feature,
                'trend_direction': trend_info['direction'],
                'trend_strength': trend_info['strength'],
                'correlation': trend_info['correlation'],
                'p_value': trend_info['p_value'],
                'percent_change': trend_info['percent_change'],
                'start_value': temporal_stats['mean'].iloc[0],
                'end_value': temporal_stats['mean'].iloc[-1]
            }
            
            trends_data.append(trend_data)
            self.trends[feature] = temporal_stats
        
        trends_df = pd.DataFrame(trends_data)
        
        # Sort by absolute correlation (strongest trends first)
        trends_df['abs_correlation'] = trends_df['correlation'].abs()
        trends_df = trends_df.sort_values('abs_correlation', ascending=False)
        
        return trends_df
    
    def _calculate_trend(self, temporal_stats: pd.DataFrame,
                        time_col: str, value_col: str) -> Dict:
        """
        Calculate trend statistics for a feature.
        
        Args:
            temporal_stats: DataFrame with temporal statistics
            time_col: Time column name
            value_col: Value column name
            
        Returns:
            Dictionary with trend information
        """
        # Convert time to numeric (years since start)
        time_numeric = temporal_stats[time_col].astype(float)
        values = temporal_stats[value_col].values
        
        # Calculate Pearson correlation
        if len(time_numeric) > 1:
            correlation, p_value = pearsonr(time_numeric, values)
        else:
            correlation, p_value = 0, 1
        
        # Determine trend direction
        if abs(correlation) < 0.1:
            direction = 'stable'
        elif correlation > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Determine trend strength
        abs_corr = abs(correlation)
        if abs_corr < 0.3:
            strength = 'weak'
        elif abs_corr < 0.7:
            strength = 'moderate'
        else:
            strength = 'strong'
        
        # Calculate percent change
        start_val = values[0] if len(values) > 0 else 0
        end_val = values[-1] if len(values) > 0 else 0
        
        if start_val != 0:
            percent_change = ((end_val - start_val) / start_val) * 100
        else:
            percent_change = 0
        
        return {
            'direction': direction,
            'strength': strength,
            'correlation': correlation,
            'p_value': p_value,
            'percent_change': percent_change
        }
    
    def compare_time_periods(self, feature: str,
                           period1: Tuple[int, int],
                           period2: Tuple[int, int],
                           temporal_column: str = 'year') -> Dict:
        """
        Compare a feature between two time periods.
        
        Args:
            feature: Feature to compare
            period1: Tuple of (start, end) for first period
            period2: Tuple of (start, end) for second period
            temporal_column: Temporal column to use
            
        Returns:
            Dictionary with comparison statistics
        """
        # Filter data for each period
        period1_data = self.df[
            (self.df[temporal_column] >= period1[0]) & 
            (self.df[temporal_column] <= period1[1])
        ][feature].dropna()
        
        period2_data = self.df[
            (self.df[temporal_column] >= period2[0]) & 
            (self.df[temporal_column] <= period2[1])
        ][feature].dropna()
        
        # Calculate statistics
        comparison = {
            'feature': feature,
            'period1': period1,
            'period2': period2,
            'period1_mean': period1_data.mean(),
            'period2_mean': period2_data.mean(),
            'period1_median': period1_data.median(),
            'period2_median': period2_data.median(),
            'period1_std': period1_data.std(),
            'period2_std': period2_data.std(),
            'mean_difference': period2_data.mean() - period1_data.mean(),
        }
        
        # Percent change
        if comparison['period1_mean'] != 0:
            comparison['percent_change'] = (
                (comparison['period2_mean'] - comparison['period1_mean']) / 
                comparison['period1_mean']
            ) * 100
        else:
            comparison['percent_change'] = 0
        
        # Statistical significance test (t-test)
        if len(period1_data) > 1 and len(period2_data) > 1:
            t_stat, p_value = stats.ttest_ind(period1_data, period2_data)
            comparison['t_statistic'] = t_stat
            comparison['p_value'] = p_value
            comparison['is_significant'] = p_value < 0.05
        else:
            comparison['t_statistic'] = None
            comparison['p_value'] = None
            comparison['is_significant'] = False
        
        return comparison
    
    def detect_breakpoints(self, feature: str,
                          temporal_column: str = 'year') -> List[int]:
        """
        Detect significant breakpoints/changes in feature trends.
        
        Args:
            feature: Feature to analyze
            temporal_column: Temporal column to use
            
        Returns:
            List of time periods where significant changes occurred
        """
        print(f"Detecting breakpoints for {feature}...")
        
        # Get temporal aggregates
        temporal_stats = self.df.groupby(temporal_column)[feature].mean().reset_index()
        values = temporal_stats[feature].values
        
        breakpoints = []
        
        # Simple breakpoint detection: look for significant slope changes
        if len(values) < 4:
            return breakpoints
        
        for i in range(1, len(values) - 1):
            # Calculate slopes before and after point
            slope_before = values[i] - values[i-1]
            slope_after = values[i+1] - values[i]
            
            # If slopes have different signs and magnitude is significant
            if (slope_before * slope_after < 0 and 
                abs(slope_before) > 0.1 * abs(values[i]) and 
                abs(slope_after) > 0.1 * abs(values[i])):
                
                period = temporal_stats[temporal_column].iloc[i]
                breakpoints.append(period)
        
        return breakpoints
    
    def calculate_feature_correlations(self, features: List[str],
                                      method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlations between features.
        
        Args:
            features: List of features to correlate
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Correlation matrix DataFrame
        """
        print(f"Calculating {method} correlations between features...")
        
        # Filter to valid features
        valid_features = [f for f in features if f in self.df.columns]
        
        if method == 'pearson':
            corr_matrix = self.df[valid_features].corr(method='pearson')
        else:
            corr_matrix = self.df[valid_features].corr(method='spearman')
        
        return corr_matrix
    
    def identify_evolving_patterns(self, 
                                  feature_groups: Dict[str, List[str]],
                                  temporal_column: str = 'year') -> Dict:
        """
        Identify how groups of related features evolve together.
        
        Args:
            feature_groups: Dictionary mapping group names to feature lists
            temporal_column: Temporal column to use
            
        Returns:
            Dictionary with evolution patterns for each group
        """
        print("Identifying evolving patterns in feature groups...")
        
        patterns = {}
        
        for group_name, features in feature_groups.items():
            print(f"  Analyzing {group_name}...")
            
            # Calculate mean trend for group
            group_trends = []
            
            for feature in features:
                if feature in self.df.columns:
                    temporal_stats = self.df.groupby(temporal_column)[feature].mean()
                    
                    # Normalize to 0-1 scale for comparison
                    if temporal_stats.max() != temporal_stats.min():
                        normalized = (temporal_stats - temporal_stats.min()) / (
                            temporal_stats.max() - temporal_stats.min()
                        )
                    else:
                        normalized = temporal_stats
                    
                    group_trends.append(normalized)
            
            if group_trends:
                # Average normalized trends
                avg_trend = pd.concat(group_trends, axis=1).mean(axis=1)
                
                # Calculate overall group trend
                time_numeric = avg_trend.index.astype(float)
                if len(time_numeric) > 1:
                    correlation, p_value = pearsonr(time_numeric, avg_trend.values)
                else:
                    correlation, p_value = 0, 1
                
                patterns[group_name] = {
                    'trend': avg_trend,
                    'correlation': correlation,
                    'p_value': p_value,
                    'direction': 'increasing' if correlation > 0.1 else (
                        'decreasing' if correlation < -0.1 else 'stable'
                    ),
                    'features': features
                }
        
        return patterns
    
    def generate_summary_report(self, features: List[str],
                              temporal_column: str = 'year') -> str:
        """
        Generate a text summary of temporal trends.
        
        Args:
            features: List of features to include
            temporal_column: Temporal column to use
            
        Returns:
            Formatted text report
        """
        trends_df = self.analyze_feature_trends(features, temporal_column)
        
        report = []
        report.append("="*70)
        report.append("PHISHING EVOLUTION TEMPORAL ANALYSIS REPORT")
        report.append("="*70)
        report.append("")
        
        # Time range
        time_range = f"{self.df[temporal_column].min()} - {self.df[temporal_column].max()}"
        report.append(f"Analysis Period: {time_range}")
        report.append(f"Total Records: {len(self.df)}")
        report.append("")
        
        # Top increasing trends
        report.append("TOP INCREASING TRENDS:")
        report.append("-" * 70)
        increasing = trends_df[trends_df['trend_direction'] == 'increasing'].head(5)
        for _, row in increasing.iterrows():
            report.append(f"  • {row['feature']}: +{row['percent_change']:.1f}% "
                        f"(correlation: {row['correlation']:.3f})")
        report.append("")
        
        # Top decreasing trends
        report.append("TOP DECREASING TRENDS:")
        report.append("-" * 70)
        decreasing = trends_df[trends_df['trend_direction'] == 'decreasing'].head(5)
        for _, row in decreasing.iterrows():
            report.append(f"  • {row['feature']}: {row['percent_change']:.1f}% "
                        f"(correlation: {row['correlation']:.3f})")
        report.append("")
        
        # Stable features
        report.append("STABLE FEATURES:")
        report.append("-" * 70)
        stable = trends_df[trends_df['trend_direction'] == 'stable'].head(5)
        for _, row in stable.iterrows():
            report.append(f"  • {row['feature']} (correlation: {row['correlation']:.3f})")
        report.append("")
        
        report.append("="*70)
        
        return "\n".join(report)


if __name__ == "__main__":
    print("Temporal Analysis Module")
    print("This module analyzes temporal evolution of phishing characteristics")