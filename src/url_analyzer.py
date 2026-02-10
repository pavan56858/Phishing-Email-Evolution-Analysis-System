"""
URL Analysis Module

Analyzes URL characteristics in phishing emails to track sophistication evolution:
- HTTPS vs HTTP usage
- Domain complexity
- Subdomain patterns
- URL length and structure
- Typosquatting detection
- Homograph attacks
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from urllib.parse import urlparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import tldextract
    import validators
except ImportError:
    print("Warning: URL analysis libraries not installed. Install with: pip install tldextract validators")


class URLAnalyzer:
    """
    Analyze URL characteristics in phishing emails for temporal evolution tracking.
    """
    
    def __init__(self):
        """Initialize the URL analyzer."""
        self.legitimate_domains = self._load_legitimate_domains()
        
    def _load_legitimate_domains(self) -> set:
        """Load common legitimate domain names."""
        return {
            'google', 'microsoft', 'apple', 'amazon', 'paypal', 'facebook',
            'twitter', 'instagram', 'netflix', 'ebay', 'yahoo', 'linkedin',
            'dropbox', 'adobe', 'salesforce', 'oracle', 'ibm', 'intel',
            'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'usbank'
        }
    
    def extract_url_features(self, df: pd.DataFrame,
                           url_column: str = 'extracted_urls') -> pd.DataFrame:
        """
        Extract comprehensive URL features from phishing emails.
        
        Args:
            df: Input DataFrame
            url_column: Column containing URLs (list of URLs)
            
        Returns:
            DataFrame with added URL features
        """
        print("Extracting URL features...")
        
        if url_column not in df.columns:
            print(f"Warning: {url_column} not found. Looking for URLs in text...")
            df = self._extract_urls_from_text(df)
        
        # Process each URL feature
        print("  - Analyzing URL structure...")
        df = self._add_url_structure_features(df, url_column)
        
        print("  - Detecting HTTPS usage...")
        df = self._add_https_features(df, url_column)
        
        print("  - Analyzing domain complexity...")
        df = self._add_domain_features(df, url_column)
        
        print("  - Detecting typosquatting...")
        df = self._add_typosquatting_features(df, url_column)
        
        print("  - Checking for suspicious patterns...")
        df = self._add_suspicious_patterns(df, url_column)
        
        print("URL feature extraction complete!")
        return df
    
    def _extract_urls_from_text(self, df: pd.DataFrame,
                               text_column: str = 'body') -> pd.DataFrame:
        """Extract URLs from text if not already extracted."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        df['extracted_urls'] = df[text_column].apply(
            lambda x: re.findall(url_pattern, str(x)) if pd.notna(x) else []
        )
        
        return df
    
    def _add_url_structure_features(self, df: pd.DataFrame,
                                   url_column: str) -> pd.DataFrame:
        """Add URL structure features."""
        
        def get_first_url(url_list):
            """Get first URL from list."""
            if isinstance(url_list, list) and len(url_list) > 0:
                return url_list[0]
            return None
        
        def url_length(url):
            """Calculate URL length."""
            return len(url) if url else 0
        
        def count_dots(url):
            """Count dots in URL."""
            return url.count('.') if url else 0
        
        def count_hyphens(url):
            """Count hyphens in URL."""
            return url.count('-') if url else 0
        
        def count_underscores(url):
            """Count underscores in URL."""
            return url.count('_') if url else 0
        
        def count_slashes(url):
            """Count slashes in URL."""
            return url.count('/') if url else 0
        
        def has_ip_address(url):
            """Check if URL contains IP address."""
            if not url:
                return 0
            # Simple IP pattern check
            ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            return 1 if re.search(ip_pattern, url) else 0
        
        # Get first URL for analysis (most prominent)
        df['first_url'] = df[url_column].apply(get_first_url)
        
        # Structure features
        df['url_length'] = df['first_url'].apply(url_length)
        df['url_dots'] = df['first_url'].apply(count_dots)
        df['url_hyphens'] = df['first_url'].apply(count_hyphens)
        df['url_underscores'] = df['first_url'].apply(count_underscores)
        df['url_slashes'] = df['first_url'].apply(count_slashes)
        df['has_ip'] = df['first_url'].apply(has_ip_address)
        
        return df
    
    def _add_https_features(self, df: pd.DataFrame,
                          url_column: str) -> pd.DataFrame:
        """Add HTTPS-related features."""
        
        def is_https(url):
            """Check if URL uses HTTPS."""
            if not url:
                return 0
            return 1 if url.startswith('https://') else 0
        
        def has_www(url):
            """Check if URL has www."""
            if not url:
                return 0
            parsed = urlparse(url)
            return 1 if 'www.' in parsed.netloc else 0
        
        df['is_https'] = df['first_url'].apply(is_https)
        df['has_www'] = df['first_url'].apply(has_www)
        
        return df
    
    def _add_domain_features(self, df: pd.DataFrame,
                           url_column: str) -> pd.DataFrame:
        """Add domain complexity features."""
        
        def extract_domain_parts(url):
            """Extract domain components using tldextract."""
            if not url:
                return None, None, None, 0
            
            try:
                extracted = tldextract.extract(url)
                subdomain = extracted.subdomain
                domain = extracted.domain
                suffix = extracted.suffix
                
                # Count subdomains
                subdomain_count = len(subdomain.split('.')) if subdomain else 0
                
                return subdomain, domain, suffix, subdomain_count
            except:
                return None, None, None, 0
        
        def domain_length(url):
            """Calculate domain length."""
            if not url:
                return 0
            
            try:
                extracted = tldextract.extract(url)
                domain = extracted.domain
                return len(domain) if domain else 0
            except:
                return 0
        
        # Extract domain parts
        domain_info = df['first_url'].apply(extract_domain_parts)
        
        df['subdomain'] = domain_info.apply(lambda x: x[0] if x else None)
        df['domain'] = domain_info.apply(lambda x: x[1] if x else None)
        df['tld'] = domain_info.apply(lambda x: x[2] if x else None)
        df['subdomain_count'] = domain_info.apply(lambda x: x[3] if x else 0)
        df['domain_length'] = df['first_url'].apply(domain_length)
        
        return df
    
    def _add_typosquatting_features(self, df: pd.DataFrame,
                                   url_column: str) -> pd.DataFrame:
        """Add typosquatting detection features."""
        
        def levenshtein_distance(s1, s2):
            """Calculate Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        def check_typosquatting(url):
            """Check if domain is similar to legitimate domains."""
            if not url:
                return 0, 0
            
            try:
                extracted = tldextract.extract(url)
                domain = extracted.domain.lower() if extracted.domain else ''
                
                if not domain:
                    return 0, 0
                
                # Check against legitimate domains
                min_distance = float('inf')
                closest_match = None
                
                for legit_domain in self.legitimate_domains:
                    distance = levenshtein_distance(domain, legit_domain)
                    if distance < min_distance:
                        min_distance = distance
                        closest_match = legit_domain
                
                # If very close (1-2 character difference), likely typosquatting
                is_typosquatting = 1 if 0 < min_distance <= 2 else 0
                
                return is_typosquatting, min_distance
            except:
                return 0, 0
        
        typo_info = df['first_url'].apply(check_typosquatting)
        
        df['is_typosquatting'] = typo_info.apply(lambda x: x[0])
        df['min_edit_distance'] = typo_info.apply(lambda x: x[1])
        
        return df
    
    def _add_suspicious_patterns(self, df: pd.DataFrame,
                                url_column: str) -> pd.DataFrame:
        """Add suspicious URL pattern features."""
        
        def has_suspicious_tld(url):
            """Check for suspicious top-level domains."""
            if not url:
                return 0
            
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']
            
            try:
                extracted = tldextract.extract(url)
                tld = '.' + extracted.suffix if extracted.suffix else ''
                return 1 if tld in suspicious_tlds else 0
            except:
                return 0
        
        def has_misleading_keywords(url):
            """Check for misleading keywords in URL."""
            if not url:
                return 0
            
            misleading_keywords = [
                'verify', 'account', 'secure', 'update', 'confirm', 'login',
                'banking', 'signin', 'validation', 'suspended', 'locked'
            ]
            
            url_lower = url.lower()
            return sum(1 for keyword in misleading_keywords if keyword in url_lower)
        
        def has_excessive_subdomains(url):
            """Check for excessive subdomains (>3)."""
            if not url:
                return 0
            
            try:
                extracted = tldextract.extract(url)
                subdomain = extracted.subdomain
                if subdomain:
                    subdomain_count = len(subdomain.split('.'))
                    return 1 if subdomain_count > 3 else 0
                return 0
            except:
                return 0
        
        def url_entropy(url):
            """Calculate entropy of URL (randomness)."""
            if not url:
                return 0
            
            from math import log2
            
            # Calculate character frequency
            char_freq = Counter(url)
            url_len = len(url)
            
            entropy = 0
            for count in char_freq.values():
                probability = count / url_len
                entropy -= probability * log2(probability)
            
            return entropy
        
        df['has_suspicious_tld'] = df['first_url'].apply(has_suspicious_tld)
        df['misleading_keywords'] = df['first_url'].apply(has_misleading_keywords)
        df['excessive_subdomains'] = df['first_url'].apply(has_excessive_subdomains)
        df['url_entropy'] = df['first_url'].apply(url_entropy)
        
        return df
    
    def analyze_url_evolution(self, df: pd.DataFrame,
                            temporal_column: str = 'year') -> pd.DataFrame:
        """
        Analyze how URL characteristics evolve over time.
        
        Args:
            df: Input DataFrame with URL features
            temporal_column: Temporal grouping column
            
        Returns:
            DataFrame with aggregated URL metrics per time period
        """
        print(f"Analyzing URL evolution over {temporal_column}...")
        
        url_metrics = [
            'is_https', 'has_www', 'subdomain_count', 'url_length',
            'domain_length', 'is_typosquatting', 'has_suspicious_tld',
            'misleading_keywords', 'url_entropy', 'has_ip'
        ]
        
        evolution_data = []
        
        for period in sorted(df[temporal_column].unique()):
            period_df = df[df[temporal_column] == period]
            
            row = {
                'period': period,
                'total_urls': len(period_df),
                'https_percentage': period_df['is_https'].mean() * 100,
                'avg_subdomain_count': period_df['subdomain_count'].mean(),
                'avg_url_length': period_df['url_length'].mean(),
                'avg_domain_length': period_df['domain_length'].mean(),
                'typosquatting_percentage': period_df['is_typosquatting'].mean() * 100,
                'suspicious_tld_percentage': period_df['has_suspicious_tld'].mean() * 100,
                'avg_misleading_keywords': period_df['misleading_keywords'].mean(),
                'avg_url_entropy': period_df['url_entropy'].mean(),
                'ip_address_percentage': period_df['has_ip'].mean() * 100
            }
            
            evolution_data.append(row)
        
        evolution_df = pd.DataFrame(evolution_data)
        
        return evolution_df
    
    def get_top_domains(self, df: pd.DataFrame,
                       temporal_column: str = 'year',
                       n_domains: int = 10) -> Dict[int, List[Tuple[str, int]]]:
        """
        Get top phishing domains per time period.
        
        Args:
            df: Input DataFrame
            temporal_column: Temporal grouping column
            n_domains: Number of top domains to extract
            
        Returns:
            Dictionary mapping time periods to top domains
        """
        print(f"Extracting top {n_domains} domains per {temporal_column}...")
        
        top_domains_by_period = {}
        
        for period in sorted(df[temporal_column].unique()):
            period_df = df[df[temporal_column] == period]
            
            # Count domain frequencies
            domains = period_df['domain'].dropna()
            domain_counts = Counter(domains)
            
            top_domains_by_period[period] = domain_counts.most_common(n_domains)
        
        return top_domains_by_period


if __name__ == "__main__":
    # Example usage
    print("URL Analyzer Module")
    print("This module analyzes URL characteristics in phishing emails")