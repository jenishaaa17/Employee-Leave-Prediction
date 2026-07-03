# IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# LOAD DATASET

abc = pd.read_csv("Employee.csv")

# EDA

print("First 5 Rows")
print(abc.head())

print("\nLast 5 Rows")
print(abc.tail())


print("\nShape")
print(abc.shape)

print("\nColumns")
print(abc.columns)

print("\nData Types")
print(abc.dtypes)

print("\nInformation")
abc.info()

print("\nStatistical Summary")
print(abc.describe())

print("\nMissing Values")
print(abc.isnull().sum())

print("\nDuplicate Values")
print(abc.duplicated().sum())

print("\nUnique Values")
print(abc.nunique())

print("\nTarget Distribution")
print(abc["LeaveOrNot"].value_counts())

print("Maximum Age:", abc["Age"].max())
print("Minimum Age:", abc["Age"].min())

# TARGET COUNTPLOT

plt.figure(figsize=(5,4))
sns.countplot(x="LeaveOrNot", data=abc)
plt.title("Employee Leave Distribution")
plt.show()

# GENDER COUNTPLOT

plt.figure(figsize=(5,4))
sns.countplot(x="Gender", data=abc)
plt.title("Gender Distribution")
plt.show()

# HISTOGRAM

abc.hist(figsize=(15,10))
plt.tight_layout()
plt.show()

# DATA CLEANING

abc.drop_duplicates(inplace=True)
abc.dropna(inplace=True)

# LABEL ENCODING

# LABEL ENCODING

education_encoder = LabelEncoder()
city_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
bench_encoder = LabelEncoder()

abc["Education"] = education_encoder.fit_transform(abc["Education"])
abc["City"] = city_encoder.fit_transform(abc["City"])
abc["Gender"] = gender_encoder.fit_transform(abc["Gender"])
abc["EverBenched"] = bench_encoder.fit_transform(abc["EverBenched"])

# SAVE ENCODERS

joblib.dump(education_encoder, "education_encoder.pkl")
joblib.dump(city_encoder, "city_encoder.pkl")
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(bench_encoder, "bench_encoder.pkl")

print("Encoders saved successfully!")

# BOXPLOT

plt.figure(figsize=(15,6))
sns.boxplot(data=abc)
plt.xticks(rotation=90)
plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(10,8))
sns.heatmap(
    abc.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# FEATURE & TARGET

X = abc.drop("LeaveOrNot", axis=1)
Y = abc["LeaveOrNot"]

# TRAIN TEST SPLIT

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

# BEFORE SMOTE

print("\nBefore SMOTE")
print(Y_train.value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x=Y_train)
plt.title("Before SMOTE")
plt.show()

# APPLY SMOTE

smote = SMOTE(random_state=42)

X_train, Y_train = smote.fit_resample(
    X_train,
    Y_train
)


# AFTER SMOTE

print("\nAfter SMOTE")
print(Y_train.value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x=Y_train)
plt.title("After SMOTE")
plt.show()

# MODEL SELECTION

model = AdaBoostClassifier(random_state=42)

# MODEL TRAINING

model.fit(X_train, Y_train)

# SAVE TRAINED MODEL

joblib.dump(model, "employee_leave_model.pkl")
print("Model saved successfully!")

# PREDICTION

Y_pred = model.predict(X_test)


# MODEL EVALUATION

print("\nAccuracy Score")
print(accuracy_score(Y_test, Y_pred))

print("\nClassification Report")
print(classification_report(Y_test, Y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(Y_test, Y_pred))

# ACTUAL VS PREDICTED

result = pd.DataFrame({
    "Actual LeaveOrNot": Y_test.values,
    "Predicted LeaveOrNot": Y_pred
})

print(result.head(10))

# CONFUSION MATRIX PLOT

cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# USER DEFINED PREDICTION

print("\nEnter Employee Details")

education = int(input("Education (Encoded Value): "))
joiningyear = int(input("Joining Year: "))
city = int(input("City (Encoded Value): "))
paymenttier = int(input("Payment Tier (1/2/3): "))
age = int(input("Age: "))
gender = int(input("Gender (Encoded Value): "))
everbenched = int(input("Ever Benched (Encoded Value): "))
experience = int(input("Experience in Current Domain: "))

employee = [[
    education,
    joiningyear,
    city,
    paymenttier,
    age,
    gender,
    everbenched,
    experience
]]

loaded_model = joblib.load("employee_leave_model.pkl")

prediction = loaded_model.predict(employee)

if prediction[0] == 1:
    print("\nPrediction : Employee is likely to Leave the Company.")
else:
    print("\nPrediction : Employee is likely to Stay in the Company.")

