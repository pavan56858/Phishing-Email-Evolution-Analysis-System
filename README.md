# Email Phishing Evolution Analyzer 🎣📊

## Overview

A novel cybersecurity analytics project that analyzes the **historical evolution of phishing email techniques** rather than performing traditional binary classification. This system studies how phishing strategies, language patterns, URL structures, and social engineering tactics have evolved over time.

## 🎯 Project Objectives

- Analyze temporal trends in phishing email characteristics
- Track evolution of language complexity and professionalism
- Study URL structure sophistication over time
- Identify emerging keywords and psychological triggers
- Visualize phishing technique evolution
- Predict future phishing patterns using historical data

## 🏗️ Architecture

```
phishing_evolution_analyzer/
├── data/                          # Datasets and processed data
│   ├── raw/                       # Original datasets
│   ├── processed/                 # Cleaned and preprocessed data
│   └── features/                  # Extracted feature datasets
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data cleaning and preparation
│   ├── temporal_analyzer.py       # Time-series analysis
│   ├── nlp_features.py            # NLP-based feature extraction
│   ├── url_analyzer.py            # URL pattern analysis
│   ├── visualization.py           # Plotting and dashboards
│   └── predictive_model.py        # Future trend prediction
├── visualizations/                # Generated charts and graphs
├── models/                        # Saved ML models
├── reports/                       # Analysis reports
└── requirements.txt               # Python dependencies
```

## 📊 Key Features

### 1. Temporal Analysis
- Year-over-year phishing technique evolution
- Quarterly trend analysis
- Seasonal pattern detection

### 2. Linguistic Evolution
- Readability score progression
- Vocabulary sophistication metrics
- Sentiment analysis over time
- Grammar quality improvement

### 3. URL Analysis
- HTTPS adoption in phishing
- Subdomain complexity trends
- Domain reputation evolution
- Typosquatting pattern changes

### 4. Social Engineering Tracking
- Urgency keyword trends
- Authority impersonation evolution
- Emotional trigger analysis
- Trust indicator usage

### 5. Predictive Analytics
- Future keyword predictions
- Technique sophistication forecasting
- Attack vector trend prediction

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Analysis**: pandas, numpy
- **NLP**: NLTK, spaCy, TextBlob
- **Machine Learning**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn, plotly
- **URL Analysis**: tldextract, validators
- **Statistical Analysis**: scipy, statsmodels

## 📈 Analysis Metrics

### Email Content Metrics
- Flesch Reading Ease Score
- Lexical diversity
- Average sentence length
- Spelling/grammar error rate
- Professional language ratio

### URL Metrics
- URL length distribution
- HTTPS vs HTTP ratio
- Subdomain count
- Top-level domain diversity
- Homograph attack prevalence

### Social Engineering Metrics
- Urgency score (time-sensitive keywords)
- Authority indicator frequency
- Fear-based language ratio
- Incentive/reward mentions

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/phishing-evolution-analyzer.git
cd phishing-evolution-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Usage

```python
from src.temporal_analyzer import PhishingEvolutionAnalyzer

# Initialize analyzer
analyzer = PhishingEvolutionAnalyzer(data_path='data/raw/phishing_emails.csv')

# Run temporal analysis
results = analyzer.analyze_evolution(start_year=2015, end_year=2024)

# Generate visualizations
analyzer.plot_language_evolution()
analyzer.plot_url_sophistication()
analyzer.plot_keyword_trends()

# Generate report
analyzer.generate_report(output_path='reports/evolution_report.html')
```

## 📊 Sample Visualizations

The project generates various visualizations including:

1. **Language Complexity Timeline** - Readability scores over time
2. **URL Sophistication Heatmap** - HTTPS usage, subdomain complexity
3. **Keyword Cloud Evolution** - Top keywords per year
4. **Social Engineering Trends** - Psychological trigger usage
5. **Attack Vector Migration** - Technique shift patterns

## 🔍 Key Findings (Sample)

Based on historical analysis (2015-2024):

- **Professionalization**: 67% increase in professional language usage
- **HTTPS Adoption**: From 12% (2015) to 78% (2024) in phishing URLs
- **Grammar Quality**: 45% reduction in spelling/grammar errors
- **Subdomain Complexity**: Average subdomains increased from 1.2 to 3.8
- **Urgency Keywords**: 23% decrease in obvious urgency words

## 🎓 Educational Value

This project demonstrates:
- Time-series analysis in cybersecurity
- NLP for security threat analysis
- Behavioral evolution tracking
- Predictive modeling for security
- Data-driven security insights

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional dataset integration
- Advanced NLP models (BERT, GPT)
- Real-time phishing trend tracking
- Multi-language phishing analysis
- Mobile phishing evolution

## 🔗 Datasets

This project uses publicly available datasets:
- [Phishing Email Dataset - Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

## 📧 Contact

For questions or collaboration:
- Email: pavankumar1292004@gmail.com

## 🙏 Acknowledgments

- APWG for phishing research resources
- Kaggle community for datasets
- Open-source NLP libraries

---

**Note**: This is an educational project. All phishing samples are used for research purposes only.
