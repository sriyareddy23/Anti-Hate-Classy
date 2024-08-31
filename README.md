# Anti-Hate Engine

## Overview

The Anti-Hate Engine is a machine learning-based classification system designed to detect and classify hate speech and spam in text data. Leveraging advanced neural network techniques, this project utilizes a Bidirectional Long Short-Term Memory (BiLSTM) model to accurately categorize input text. The system preprocesses text data using Regex Filtering and Lemmatizing, and evaluates performance with standard metrics such as accuracy, precision, recall, and F1-score.

## Tech Stack

- **Programming Language**: Python
- **Libraries**: TensorFlow, Keras, NumPy
- **Preprocessing**: Regex Filtering, Lemmatizing
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score

## Features

- **BiLSTM Model**: Utilizes a Bidirectional LSTM architecture to handle long-term dependencies in sequential data.
- **Data Preprocessing**: Cleans and normalizes text using Regex Filtering and Lemmatizing to improve model accuracy.
- **Performance Evaluation**: Measures effectiveness using accuracy, precision, recall, and F1-score, addressing class imbalance issues.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sriyareddy23/anti-hate-engine.git
    ```

## Usage

1. Preprocess text data:
    ```python
    python preprocess.py
    ```
2. Train the model:
    ```python
    python train_model.py
    ```
3. Evaluate the model:
    ```python
    python evaluate_model.py
    ```

## Results

- **Hate Speech Classification Accuracy**: 99.26%
- **Spam Classification Accuracy**: 76.49%


### - [Docs] -
(sic.) We have here a mélange of the source code with, a pinch of salt at the ready, "*fun*" explanations. 
### - [Spam Model] -
The spam model lives here.
### - [Hate Model] -
The [hate model](https://drive.google.com/file/d/1bTSPMLzol0Blo4ona6IJbeN4tMe91pAT/view?usp=sharing) does not, in truth, live here; this is its vacation destination and in fact a summer house.
### - [Tokenizers] -
These convert text to vectors.
### - [Main] -
snek lang go brr
