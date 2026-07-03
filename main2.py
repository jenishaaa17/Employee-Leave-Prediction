#Data Collection

import pandas as pd

abc = pd.read_csv("Employee.csv")

print(abc)
print(abc.tail(10))
print(abc.shape)
print(abc.columns)
print(abc.dtypes)
abc.info()
print(abc.describe())
print("Maximum Age:", abc["Age"].max())
print("Minimum Age:", abc["Age"].min())

#Data Cleaning

print(abc.isnull())
print(abc.isna())

print(abc.duplicated())
print(abc[abc.duplicated()])
print("Duplicate Count:", abc.duplicated().sum())

abc = abc.drop_duplicates()

#Label Encoding

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

abc["Education"] = le.fit_transform(abc["Education"])
abc["City"] = le.fit_transform(abc["City"])
abc["Gender"] = le.fit_transform(abc["Gender"])
abc["EverBenched"] = le.fit_transform(abc["EverBenched"])

print(abc)

#Train Test Split

from sklearn.model_selection import train_test_split

X = abc.drop("LeaveOrNot", axis=1)
Y = abc["LeaveOrNot"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

#SMOTE

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, Y_train = smote.fit_resample(X_train, Y_train)

print("After SMOTE")

print("X_train Shape:", X_train.shape)
print("Y_train Shape:", Y_train.shape)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create AdaBoost model
model = AdaBoostClassifier(random_state=42)

# Train the model
model.fit(X_train, Y_train)

# Predict on test data
Y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Actual vs Predicted Table
result = pd.DataFrame({
    "Actual LeaveOrNot": Y_test.values,
    "Predicted LeaveOrNot": Y_pred
})

print(result.head(10))

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(Y_test, Y_pred))

# Classification Report
print("\nClassification Report")
print(classification_report(Y_test, Y_pred))
