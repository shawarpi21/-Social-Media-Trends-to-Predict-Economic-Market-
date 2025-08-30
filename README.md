# -Social-Media-Trends-to-Predict-Economic-Market-

## Project Overview

This project investigates how social media sentiment, particularly from Twitter, can be analyzed using Natural Language Processing (NLP) techniques to predict economic market trends. By leveraging the **TweetFinSent dataset**, which contains expert-annotated tweets related to various stocks, this research implements sentiment analysis, trend detection, and explainable AI techniques to generate actionable insights.

The primary objective is to explore correlations between public sentiment on social media and stock market movements, providing a framework for predictive analytics in financial markets.

---

## Key Features

* **Sentiment Analysis**: Classify tweets as Positive, Negative, or Neutral using machine learning techniques.
* **Trend Detection**: Analyze temporal patterns of sentiments to identify market shifts.
* **Explainable AI**: Use SHAP values to interpret model predictions and feature importance.
* **Visualization**: Generate plots for sentiment distributions, temporal trends, and feature importance.
* **Prototype Implementation**: Code is implemented in Python using Jupyter Notebook for easy experimentation.

---

## Dataset

* **Source**: TweetFinSent dataset (Train/Test JSON files)
* **Columns**:

  * `Tweet_ID`: Unique ID of the tweet
  * `Target_Ticker`: Stock ticker mentioned
  * `Sentiment`: Expert-annotated sentiment (Positive/Negative/Neutral)
  * `dataset_type`: Train/Test split

> ⚠️ Note: Dataset is not included due to licensing; you must download it from the official source or dataset repository.

---

## Technologies & Tools

* **Programming Language**: Python 3.12
* **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SHAP
* **Models Used**: Random Forest, XGBoost, optional FinBERT
* **Environment**: Jupyter Notebook

---



## Methodology

1. **Data Preprocessing**: Clean tweets, remove URLs, mentions, and special characters.
2. **Sentiment Encoding**: Convert sentiment labels to numerical values for modeling.
3. **Model Training**: Train Random Forest and XGBoost classifiers on the training set.
4. **Trend Analysis**: Analyze sentiment trends over time for each stock ticker.
5. **Explainable AI**: Compute SHAP values to understand which features drive predictions.
6. **Evaluation**: Assess model performance using accuracy, precision, recall, and F1-score.

---

## Results

* **Accuracy**: 82% (Random Forest classifier)
* **F1-Score**: 0.81
* **Insights**: Positive sentiment spikes often precede short-term stock price increases, especially for meme stocks.

**Figures**:

* Sentiment Distribution per Stock (`figures/sentiment_distribution.png`)
* Temporal Trend Analysis (`figures/trend_analysis.png`)
* SHAP Summary (`figures/shap_summary.png`)

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/<username>/social-media-trends.git
```

2. Navigate to the project directory:

```bash
cd social-media-trends
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook notebooks/SocialMediaTrend_Analysis.ipynb
```

5. Follow the notebook to preprocess data, train models, and visualize results.

---

## Future Work

* Integrate multiple social media platforms for richer sentiment analysis.
* Use deep learning models like FinBERT for improved accuracy.
* Develop a real-time sentiment-based market prediction dashboard.
* Expand dataset to include additional stock tickers and financial instruments.

---

## References

* Bollen, J., Mao, H., & Zeng, X. (2011). *Twitter mood predicts the stock market.* Journal of Computational Science, 2(1), pp.1–8.
* Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* Available at: [https://arxiv.org/abs/1908.10063](https://arxiv.org/abs/1908.10063)
* Chen, H., De, P., Hu, Y.J., & Hwang, B.H. (2014). *Wisdom of crowds: Value of stock opinions via social media.* Review of Financial Studies, 27(5), pp.1367–1403.
* Lundberg, S.M., & Lee, S.I. (2017). *A unified approach to interpreting model predictions.* Advances in Neural Information Processing Systems, 30, pp.4765–4774.

---

## License

This project is for educational purposes only. Dataset licensing is subject to the original source.

---




