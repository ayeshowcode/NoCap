# Fake News Detection Project

A machine learning project that classifies news articles as either **FAKE** or **REAL** using Support Vector Machine (SVM) with TF-IDF vectorization.

## 📊 Dataset

The project uses `fake_or_real_news.csv` containing news articles with binary labels:
- **FAKE** (converted to 1)
- **REAL** (converted to 0)

## 🧠 Model Architecture

### Features
- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Algorithm**: Linear Support Vector Classifier (LinearSVC)
- **Train/Test Split**: 80%/20% with random_state=42

### Preprocessing Pipeline
1. Load dataset from CSV
2. Convert text labels to binary (FAKE=1, REAL=0)
3. Split features (text) and targets (labels)
4. Apply TF-IDF vectorization to convert text to numerical features
5. Train Linear SVM classifier

## 📈 Results

### Model Performance
- **Overall Accuracy**: **93.84%** on test set
- **Test Set Size**: 1,267 samples
- **Correct Predictions**: 1,189 out of 1,267
- **Error Rate**: 6.16% (78 incorrect predictions)

### Sample Predictions
The model was tested on individual samples:

#### Sample #10 (Incorrect Prediction)
- **Model Prediction**: REAL (0)
- **Actual Label**: FAKE (1)
- **Result**: ❌ **WRONG**

#### Sample #20 (Correct Prediction)
- **Model Prediction**: Matches actual label
- **Result**: ✅ **RIGHT**

## 🔧 Implementation Details

### Libraries Used
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
```

### Key Code Structure
```python
# Data preprocessing
df['fake'] = df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)
X, y = df['text'], df['fake']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)
```

## 🎯 Model Evaluation

### Prediction Process
1. Transform new text using the trained TF-IDF vectorizer
2. Apply the trained SVM classifier
3. Output: 0 (REAL) or 1 (FAKE)

### Testing Individual Articles
The model can predict on new text by:
```python
# Vectorize new text
vectorized_text = vectorizer.transform([new_text])
# Make prediction
prediction = clf.predict(vectorized_text)
```

## 📁 Project Structure
```
fake news detection/
├── main.ipynb              # Main Jupyter notebook with full pipeline
├── fake_or_real_news.csv   # Dataset
├── mytext.txt              # Sample text file for testing
└── README.md               # This file
```

## 🚀 Usage

1. **Load the notebook**: Open `main.ipynb` in Jupyter/VS Code
2. **Run all cells**: Execute the complete pipeline from data loading to evaluation
3. **Test new articles**: Use the last few cells to test individual articles
4. **View results**: Model achieves 93.84% accuracy on unseen data

## 📊 Performance Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | 93.84% |
| **Total Test Samples** | 1,267 |
| **Correct Predictions** | 1,189 |
| **Incorrect Predictions** | 78 |
| **Model Type** | Linear SVM |
| **Feature Engineering** | TF-IDF Vectorization |

## 🎓 Key Insights

- **High Accuracy**: The model demonstrates strong performance with >93% accuracy
- **Text Classification**: Successfully converts unstructured text to numerical features
- **Practical Application**: Can classify individual news articles in real-time
- **Error Analysis**: Some misclassifications occur, showing room for improvement

## 🔮 Future Improvements

- Implement ensemble methods
- Add text preprocessing (stopword removal, stemming)
- Try different algorithms (Random Forest, Neural Networks)
- Feature engineering (article metadata, linguistic features)
- Cross-validation for robust evaluation

---

*This project demonstrates effective text classification for fake news detection using traditional machine learning techniques.*
