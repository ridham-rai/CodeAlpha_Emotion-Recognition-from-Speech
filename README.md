# CodeAlpha_Emotion-Recognition-from-Speech
This repository contains a deep learning model that classifies emotions in speech audio, identifying emotions like happiness, anger, and sadness from spoken sentences. This project focuses on building a deep learning model to recognize emotions in speech. The model leverages speech processing and deep learning techniques to classify spoken sentences into different emotional categories. Emotions such as anger, happiness, and sadness are recognized from audio recordings, aiming to improve human-computer interaction, speech-based applications, and emotion-sensitive technologies.

# Model Overview
Emotion recognition from speech is an exciting and challenging task that involves analyzing various aspects of speech such as tone, pitch, tempo, and rhythm. These characteristics can convey emotional states in the speaker’s voice. The goal of this project is to train a model that can automatically detect these emotional cues in speech and categorize them into predefined classes.

To accomplish this, the model uses Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), which are effective in processing sequential data, such as audio signals. The model is trained on a speech dataset containing recordings of target words spoken with three distinct emotions: anger, happiness, and sadness. The dataset is used to extract audio features, such as Mel-frequency cepstral coefficients (MFCCs), which capture the spectral features of the speech signal.

# Key Components:
# Data Preprocessing:
The raw audio files are processed to extract relevant features (e.g., MFCCs) that represent the emotional content of the speech. These features are then normalized to ensure the model can efficiently learn from them.
# Model Architecture:
The model uses a combination of CNNs to extract spatial features from the input audio spectrogram and RNNs, such as LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit), to capture the temporal dependencies between speech frames. The model outputs a probability distribution over the emotional classes.
# Training:
The model is trained on the audio features using a softmax activation function in the output layer, which classifies the input speech into one of the three emotions: anger, happiness, or sadness.
# Evaluation:
After training, the model is evaluated using test data. Performance is assessed using metrics like accuracy, precision, and recall.
# Use Cases:
Human-Computer Interaction: Improving virtual assistants like Siri, Alexa, or Google Assistant to understand the emotional context of a user's voice.
Call Center Automation: Detecting customer emotions for better service quality.
Healthcare: Recognizing emotional distress in patients to provide timely intervention.
Entertainment and Media: Enhancing interactive experiences by adapting to the emotional tone of users.
# Dataset Information
The dataset used for this project consists of 2800 audio recordings of 200 target words spoken in the carrier phrase "Say the word _". These recordings were made by two actresses, aged 26 and 64, portraying three different emotions: anger, happiness, and sadness.

The dataset contains speech data in the WAV format, and each recording corresponds to one of the three emotions spoken in the context of the target word. The data is organized into subfolders corresponding to each emotion, with separate folders for the two actresses. The diversity in speech patterns (age and emotional variation) helps the model generalize well to different speakers and emotional contexts.

# Libraries Used
TensorFlow: For creating and training the deep learning model.
Keras: High-level neural networks API for building and training deep learning models.
Librosa: A Python package for audio processing and feature extraction.
NumPy: For handling numerical data and arrays.
Matplotlib: For visualizing training and evaluation metrics (e.g., accuracy and loss curves).
Scikit-learn: For splitting data and evaluating the model's performance.
