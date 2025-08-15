**Level 5 README – Audio & Image Advanced Projects**
----------------------------------------------------

🎙️ Level 5 Machine Learning Projects – Audio & Advanced Image Classification
=============================================================================

This repository highlights advanced ML projects involving audio-based classification and image classification with transfer learning. These projects were developed in the final stage (Level 5) of the 5-level ML roadmap, combining deep learning techniques for different data modalities.

📁 Projects
-----------

### 1\. Speech Emotion Recognition (CSV Features)

*   **Objective:** Predict emotions from speech using pre-extracted audio features.
    
*   **Data:** CSV file containing MFCC aggregates, chroma, and spectral features.
    
*   **Target:** Emotion label (e.g., happy, sad, angry, neutral)
    
*   **Best Model:** Random Forest / SVM with cross-validation
    
*   **Key Skills:** Feature scaling, label encoding, classical ML on structured audio data.
    

### 2\. Speech Emotion Recognition (Actors’ Audio)

*   **Objective:** Classify emotions from raw audio recordings of actors.
    
*   **Data:** RAVDESS-like dataset with WAV audio files labeled with emotions.
    
*   **Target:** Emotion category (7 classes)
    
*   **Best Model:** CNN over MFCC spectrograms (with dropout & early stopping)
    
*   **Key Skills:** MFCC extraction, CNN audio classification, data augmentation (time-stretch, pitch-shift).
    

### 3\. Image Classification with Transfer Learning (ResNet50)

*   **Objective:** Classify natural scene images into 6 categories.
    
*   **Data:** Intel Image Classification Dataset
    
*   **Target:** Scene category (buildings, forest, glacier, mountain, sea, street)
    
*   **Best Model:** ResNet50 (frozen backbone + custom dense head) – Test Accuracy ~62.9%
    
*   **Key Skills:** Transfer learning, data augmentation, fine-tuning strategies.
    

### 4\. Fake News Detection

*   **Objective:** Classify news articles as fake or true.
    
*   **Data:** Combined fake.csv and true.csv datasets.
    
*   **Target:** 0 (Fake) or 1 (True)
    
*   **Best Model:** Logistic Regression with TF-IDF features – Accuracy ~98.5%
    
*   **Key Skills:** Text preprocessing, TF-IDF vectorization, logistic regression, high-accuracy NLP classification.
    

🧰 Tools & Techniques
---------------------

*   Python, TensorFlow/Keras, Librosa
    
*   Audio feature extraction (MFCC, Chroma, Spectrograms)
    
*   CNN architectures for audio/image classification
    
*   Transfer learning with ResNet50
    
*   Classical ML models for structured features
    
*   NLP text vectorization (TF-IDF)
    
*   Model evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix
    

🏁 Outcomes
-----------

*   Gained experience with multi-modal ML tasks (audio, image, text).
    
*   Learned transfer learning strategies and audio feature processing.
    
*   Built high-performing models for classification tasks across different data types.
    

📝 Author
---------

Bishal Pandey[My Portfolio](https://pandeybishal921.wixsite.com/my-site)