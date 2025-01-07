
# Sentiment Analysis with Transformer Models

This repository contains a Python  for performing sentiment analysis on a  dataset using transformer models, including DistilBERT. 

## Dataset Overview

The dataset consists of labeled with one of four sentiment categories:

- **Negative**: Indicates unfavorable sentiment.
- **Positive**: Indicates favorable sentiment.

### Data Columns
- **Sentiment**: The sentiment expressed in the tweet (Negative, Positive).
- ** Content**: The actual text of the tweet.

### Sentiment Distribution
- **Negative**: 99.92% of the dataset.
- **Positive**: 99.98% of the dataset.

## Models Implemented

The Python uses the following transformer models for sentiment analysis:
- **DistilBERT**: A smaller, faster version of BERT.

## How to Use

### Step 1: Load the Dataset
The Python loads the dataset containing  tweet content. The dataset is pre-split into sentiment and score.

### Step 2: Preprocess the Data
The Python includes a preprocessing step that:
- Tokenizes the tweet text using the appropriate tokenizer for each transformer model.
- Converts sentiment labels into numerical format for model training.

### Step 3: Train the Model
The python allows you to choose from the following transformer models:
- **DistilBERT**


### Step 4: Evaluate the Model
Once trained, the model is evaluated on the validation set.
- **Accuracy**: Prportion of correct predictions.
-**sentiment**: Prportion of correct predictions


### Step 5: Visualize Results
The python includes predictions sentiment, score.

```python
model.save_pretrained('model_dir')
tokenizer.save_pretrained('model_dir')
```

## Example Usage

Here’s a simple example of how to run the python:

1. Load and preprocess the dataset.
2. uncased_Finetune a transformer model (e.g.,distilbert-base-uncased-finetuned-sst-2-english ).
3. Evaluate the model and visualize performance sentiment and score.
4. Save the trained model for future use.
