# 🧠 Real ML Roadmap – Machine Learning Projects (Level 1 to 5)

This repository documents my journey through the [5-Level ML Project Roadmap](https://www.youtube.com/watch?v=Bx4BYXOE9SQ) by Marina Wyss.  
Each level builds upon the previous one — starting from simple classical ML tasks to advanced multi-modal deep learning projects.

---

## 📊 Project Progress Overview

| Level  | Focus Area                                           | Status        |
|--------|------------------------------------------------------|---------------|
| 1️⃣ Level 1 | Beginner ML – Small Classical Models               | ✅ Completed |
| 2️⃣ Level 2 | Real-World Regression – Tabular Data                | ✅ Completed |
| 3️⃣ Level 3 | Classification on Tabular Data                      | ✅ Completed |
| 4️⃣ Level 4 | Computer Vision – CNNs for Image Classification     | ✅ Completed |
| 5️⃣ Level 5 | Audio, Transfer Learning & Multi-Modal Projects     | ✅ Completed |

---

## ✅ Level 1 – Beginner Projects

> Small, structured datasets using classical ML models

| 📄 Project                        | 📦 Description                                  | ⚙️ Algorithm               | 🎯 Accuracy |
|----------------------------------|------------------------------------------------|----------------------------|-------------|
| Iris Flower Classification       | Classify flowers into 3 species based on petal/sepal size | Logistic Regression      | 100%        |
| MNIST Digit Classifier (8×8)     | Recognize handwritten digits from small grayscale images   | Logistic Regression      | ~97.2%      |
| Breast Cancer Detection          | Predict if a tumor is benign or malignant based on biopsy data | Logistic Regression + Balanced Weights | ~95.6%      |

**Skills Gained:**  
- Data preprocessing (scaling, splitting, cleaning)  
- Handling class imbalance (`class_weight='balanced'`)  
- Confusion matrix visualization  
- Multi-class & binary classification  

---

## ✅ Level 2 – Real-World Regression Projects

> Structured/tabular datasets with pipelines, feature engineering, and evaluation metrics for regression.

| 📄 Project                        | 📦 Description                                  | ⚙️ Algorithm               | 📏 Metrics |
|----------------------------------|------------------------------------------------|----------------------------|------------|
| California Housing Price Prediction | Predict median house prices from location & demographics | Random Forest (tuned) | RMSE: 0.50, R²: 0.8058 |
| Bike Sharing Demand Prediction  | Predict daily bike rentals based on weather & season | Random Forest (tuned) | RMSE: 73.27, R²: 0.8374 |
| Medical Insurance Cost Prediction | Predict insurance charges based on personal attributes | Random Forest (tuned) | RMSE: 4076.57, R²: 0.8742 |

**Skills Gained:**  
- Feature engineering (temporal, categorical encoding)  
- Evaluation metrics (MAE, RMSE, R²)  
- Hyperparameter tuning with `GridSearchCV`  
- Handling skewed data distributions  

---

## ✅ Level 3 – Classification on Tabular Data

> Binary & multiclass classification tasks with structured datasets.

| 📄 Project                        | 📦 Description                                  | ⚙️ Algorithm               | 🎯 Accuracy |
|----------------------------------|------------------------------------------------|----------------------------|-------------|
| Titanic Survival Prediction     | Predict passenger survival on the Titanic | Random Forest | ~82.68% |
| Heart Disease Prediction        | Predict presence of heart disease | Random Forest | ~87.5% |
| Loan Approval Prediction        | Predict loan application approval | Voting Classifier (Soft Voting) | ~84.4% |

**Skills Gained:**  
- Feature engineering (Title extraction, imputation)  
- One-hot & label encoding  
- Ensemble models (Random Forest, Voting Classifiers)  
- Pipelines for preprocessing & scaling  

---

## ✅ Level 4 – Computer Vision Projects (CNNs)

> Image classification using custom CNNs on binary and multiclass datasets.

| 📄 Project                        | 📦 Description                                  | ⚙️ Algorithm               | 🎯 Accuracy |
|----------------------------------|------------------------------------------------|----------------------------|-------------|
| Cat vs Dog Classification       | Classify images as cat or dog | Custom CNN | ~81% |
| CIFAR-10 Image Classification   | Classify 10 object categories | Custom CNN + Augmentation | ~73.7% |
| Flower Classification           | Classify flowers into 5 classes | CNN + Augmentation + EarlyStopping | ~92% |

**Skills Gained:**  
- Image preprocessing & augmentation  
- CNN architecture design  
- Binary & multiclass image classification  
- Visual performance evaluation (curves, confusion matrix)  

---

## ✅ Level 5 – Audio & Advanced Image/Text Projects

> Multi-modal ML — audio, images (transfer learning), and NLP text classification.

| 📄 Project                        | 📦 Description                                  | ⚙️ Algorithm               | 📏 Metrics |
|----------------------------------|------------------------------------------------|----------------------------|------------|
| Speech Emotion Recognition (CSV Features) | Classify emotions from pre-extracted audio features | Random Forest / SVM | High CV stability |
| Speech Emotion Recognition (Audio Files) | Classify emotions from raw actor speech (MFCC features) | CNN over MFCC spectrograms | Robust accuracy |
| Image Classification (Transfer Learning) | Classify natural scenes into 6 categories | ResNet50 (frozen backbone) | ~62.9% |
| Fake News Detection (NLP)        | Classify news as fake or true using TF-IDF | Logistic Regression | 98.54% Accuracy |

**Skills Gained:**  
- Audio feature extraction (MFCC, Chroma)  
- Transfer learning & fine-tuning (ResNet50)  
- Text preprocessing & TF-IDF vectorization  
- Multi-modal deep learning workflows  

---

## 🛠️ Tools & Libraries Across All Levels

- **Languages:** Python  
- **ML/DL:** scikit-learn, TensorFlow/Keras  
- **Data Processing:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Audio Processing:** librosa  
- **NLP:** scikit-learn (TF-IDF), NLTK  
- **Workflow:** Pipelines, GridSearchCV, EarlyStopping, Model Saving (`joblib`, `h5`)  

---

## 📚 Key Takeaways

- Built **15+ ML projects** across tabular, image, audio, and text data.  
- Progressed from **classical ML** → **deep learning** → **multi-modal AI**.  
- Learned to design complete ML pipelines, handle diverse datasets, and apply both **custom architectures** and **transfer learning**.  

---

## 📌 Author

**Bishal Kumar Pandey**  
🎯 Aspiring Machine Learning Engineer | Learning by building  
🌍 [Portfolio Website](https://pandeybishal921.wixsite.com/my-site)  

---

## 🌟 Support the Journey

If this inspires you or helps in your learning, give it a ⭐ and share!  
