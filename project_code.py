import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# DATA PREPROCESSING

df = pd.read_csv('breast-cancer.csv')
df = df.drop(columns=['id'])
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

y = df['diagnosis']  
X = df.drop(columns=['diagnosis'])  

# Remove nulls and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Remove outliers using the IQR method
column_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

for column_name in column_names:
  Q1 = df[column_name].quantile(0.25)
  Q3 = df[column_name].quantile(0.75)
  IQR = Q3 - Q1
  df = df[~((df[column_name] < (Q1 - 1.5 * IQR)) | (df[column_name] > (Q3 + 1.5 * IQR)))]

# Standardization
scaler = StandardScaler()
df[column_names] = scaler.fit_transform(df[column_names])

# Normalization
normalizer = MinMaxScaler()
df[column_names] = normalizer.fit_transform(df[column_names])



# FEATURE LEARNING

# Calculate correlation with the target variable
correlation_matrix = X.corrwith(y)
selected_features = correlation_matrix[correlation_matrix.abs() > 0.5].index.tolist()
X_selected = X[selected_features]
print("Selected Features:", selected_features)



# MODEL TRAINING
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size = 0.2, random_state=42)
svm = SVC(kernel='rbf', random_state=42)

# Cross validation (5-fold)
cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
print("Cross-validation scores: ", cv_scores)
print("Average cross validation score: ", np.mean(cv_scores))

# Train (fit) SVM model on training data
svm.fit(X_train, y_train)

# Predictions
y_pred_train = svm.predict(X_train)



# MODEL TESTING
y_pred_test = svm.predict(X_test)



# MODEL PERFORMANCE EVALUATION
# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("Classification Report: \n", classification_report(y_test, y_pred_test))

#----------------------------------------------------------------------------------------------------------------------------------#

# USER INPUT FOR PREDICTION
user_file = input('Enter a CSV file: ')
user_input_df = pd.read_csv(user_file)
user_input_df = user_input_df.drop(columns=['id']) 
X_user_input = user_input_df[selected_features] 

# Remove nulls and duplicates
user_input_df.dropna(inplace=True)
user_input_df.drop_duplicates(inplace=True)

# Remove outliers using the IQR method
column_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

# Standardization
scaler = StandardScaler()
user_input_df[column_names] = scaler.fit_transform(user_input_df[column_names])

# Normalization
normalizer = MinMaxScaler()
user_input_df[column_names] = normalizer.fit_transform(user_input_df[column_names])

# Make a prediction
prediction = svm.predict(X_user_input)

# Display the prediction
if prediction[0] == 1:
    print("The model predicts the tumor is Malignant.")
else:
    print("The model predicts the tumor is Benign.")

