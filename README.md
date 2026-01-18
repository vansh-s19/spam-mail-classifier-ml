# Spam Mail Prediction using Machine Learning

This project is a **Spam Mail Classification system** built using **Machine Learning (Logistic Regression)** and **Natural Language Processing (TF-IDF Vectorization)**.  
The model classifies an email message as **Spam** or **Ham (Not Spam)** based on its textual content.

---

## Project Overview

Email spam detection is a classic **binary classification problem** in Machine Learning.  
In this project:
- Email text is converted into numerical features using **TF-IDF Vectorizer**
- A **Logistic Regression** model is trained on labeled email data
- Users can input **multi-line email content via the terminal**
- The model predicts whether the email is **Spam or Ham**

---

## Dataset

- Dataset used: `mail_data.csv`
- Columns:
  - `Category` → `spam` or `ham`
  - `Message` → email content
- Labels:
  - `spam` → `0`
  - `ham` → `1`

---

## Technologies Used

- Python 3
- NumPy
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

---

## Project Structure

Spam Mail Prediction/
│
├── data/
│   └── mail_data.csv
│
├── model/
│   ├── Spam_Mail_Prediction.ipynb
│   └── spam_mail_prediction.py
│
├── requirements.txt
└── README.md

---

## How the Model Works

1. Load and preprocess the dataset
2. Encode labels (`spam = 0`, `ham = 1`)
3. Split data into training and testing sets
4. Convert text data into numerical vectors using **TF-IDF**
5. Train a **Logistic Regression** classifier
6. Evaluate accuracy on training and testing data
7. Accept user input and predict spam/ham

---

## How to Run the Project

### 1. Clone the Repository
git clone 
cd Spam-Mail-Prediction

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Script
python spam_mail_prediction.py

---

## Multi-line Email Input (CLI Mode)

- Paste the **entire email content** (multiple lines supported)
- Press **ENTER on an empty line** to get the prediction
- Type **EXIT** to stop the program

Example: Paste the email content below.
Press ENTER on an empty line to run prediction.

Congratulations! You have won a free gift.
Click the link to claim now.

Prediction: Spam mail

---

## Model Performance

- Accuracy is printed for:
  - Training data
  - Testing data
- Logistic Regression performs well for text-based binary classification problems

---

## Future Improvements

- Save and load trained model using `joblib`
- Show prediction confidence score
- Deploy using Flask or Streamlit
- Improve preprocessing (lemmatization, n-grams)

---

## Author

**Vansh Saxena**  
Machine Learning & DSA Enthusiast  

---

## License

This project is for **learning and educational purposes**.