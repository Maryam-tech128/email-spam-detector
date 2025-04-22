Overview
This project is a machine learning-based spam detection system that classifies emails as either spam or ham (not spam). It uses Natural Language Processing (NLP) techniques along with machine learning models to analyze and classify email text data.

Workflow

1. Data Collection
The dataset used consists of labeled emails as spam or ham.
Text data was extracted and used for feature engineering and model training.

2. Text Preprocessing
Converted all text to lowercase.
Removed punctuation, special characters, and stopwords.
Applied stemming to normalize words.
Transformed text into numerical format using TF-IDF Vectorization.

3. Exploratory Data Analysis (EDA)
To understand data distribution and spam characteristics:
Pie Chart for class distribution.
Bar Plot for top spam words.
Pair Plot to observe relationships.
Heatmap to show correlation between features.
Histogram Plot for text length and word count.

4. Feature Engineering
Additional features were extracted to enrich the dataset:
Message length
Number of words
Number of links
Number of uppercase character
These features were combined with TF-IDF vectors to improve model performance.

5. Model Building
Several machine learning models were trained and evaluated:
Gaussian Naive Bayes (GNB)
Multinomial Naive Bayes (MNB)
Bernoulli Naive Bayes (BNB)
Voting Classifier (Ensemble of all 3)

6. Model Evaluation
Performance was evaluated using:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
  
7. Output
The best-performing model (Voting Classifier) is able to classify email messages with high accuracy.

Tools & Libraries
Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
TfidfVectorizer
VotingClassifier
