# Stock Market Volatility Prediction

## Project Overview
This project explores machine learning approaches to predict stock market volatility, focusing specifically on the SPY ETF that tracks the S&P 500 index. By leveraging historical price data and technical indicators, I've developed models that forecast future volatility with measurable accuracy.


## Key Features
- Historical data analysis of SPY ETF from 2010 to 2025
- Comprehensive feature engineering combining technical indicators and volatility metrics
- Implementation of multiple machine learning models (Random Forest, XGBoost, LSTM neural networks)
- Walk-forward validation methodology for realistic performance assessment
- Seasonal volatility analysis and market regime identification
- Model performance evaluation across different market conditions

## Repository Structure
```
├── doc/                # Documentation files
│   ├── blog.qmd        # Quarto source file for the blog post
│   ├── blog.html       # Rendered HTML blog post
│   └── references.bib  # Bibliography references
├── src/                # Source code for the project
├── results/            # Model outputs and evaluation metrics
├── capstone_data/      # Stored datasets (excluded from version control)
├── environment.yml     # Conda environment specification
├── index.html          # Website redirect to the blog post
└── README.md           # This file
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- Conda or Miniconda (recommended for environment management)
- Quarto (for rendering the blog post)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/KIzadi/capstone.git
cd capstone


conda env create -f environment.yml
conda activate capstone


pip install yfinance tensorflow xgboost
```

### Data Collection
The project uses the `yfinance` API to download historical SPY and VIX data. The data collection script is included in the blog post.

## Viewing the Blog Post
The blog post is available online at [https://kizadi.github.io/capstone/doc/blog.html](https://kizadi.github.io/capstone/doc/blog.html).

To render the blog post locally:
```bash
quarto render doc/blog.qmd
```
Then open `doc/blog.html` in your web browser.

## Model Implementation
The project implements several machine learning models:
- Random Forest with optimized hyperparameters
- XGBoost gradient boosting 
- LSTM neural networks for sequence modeling
- Linear regression (baseline model)

Each model was evaluated using a walk-forward validation approach to prevent look-ahead bias and ensure realistic performance assessment.

## Results and Findings
The models demonstrated a 12-18% improvement over traditional statistical methods when measured by RMSE. Detailed performance metrics, feature importance analysis, and visualization of predictions are available in the blog post.

Key findings include:
- Tree-based models (Random Forest, XGBoost) provided the best balance of performance and interpretability
- The VIX index and recent volatility measures were the most predictive features
- Models performed better during moderate and high volatility periods than during extremely low volatility
- A persistent mean reversion bias was observed in model predictions
