import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""### Data collection and Preprocessing"""

raw_mail_data = pd.read_csv('/Users/vanshsaxena/Documents/Machine Learning Models/Spam Mail Prediction/data/mail_data.csv')

#replace all the null values to null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

"""### Label Encoding"""

# Label Spam mail as 0, and Ham mail as 1

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

# Separating the data as text and Labels
X = mail_data['Message']
Y = mail_data['Category']

"""### Spliting the data into training and testing"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

"""### Feature Extraction"""

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

#converting Y_train and Y_test values as Integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

"""### Training the Model"""

model = LogisticRegression()

model.fit(X_train_features, Y_train )

"""#### Evaluating the trained model"""

# Prediction on training model
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy on training data : ', accuracy_on_training_data)

# Prediction on testing model
prediction_on_testing_data = model.predict(X_test_features)
accuracy_on_testing_data = accuracy_score(Y_test, prediction_on_testing_data)

print('Accuracy on testing data: ', accuracy_on_testing_data)

"""## Building a Predictive System"""

# Take mail content from the user
input_mail = input("Enter the mail content to classify as Spam or Ham:\n")

# convert text to feature vectors
input_data_features = feature_extraction.transform([input_mail])

# making prediction
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')