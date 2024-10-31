import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
#Improving accuracy
from imblearn.over_sampling import SMOTE

df = pd.read_csv('churn.csv')

sns.set_style(style="whitegrid")
plt.figure(figsize=(12, 10))

#sns.countplot(x='Exited', data=df)
plt.title('Churn Distribution')
#sns.histplot(data=df, x='Age', kde=True)
plt.title('Age Distribution')

#sns.scatterplot(data=df, x='CreditScore', y='Age', hue='Exited')
plt.title('Credit Score vs Age')

#sns.boxplot(data=df, x='Exited', y='Balance')
plt.title('Balance vs Churn')

#sns.boxplot(x='Exited', y='CreditScore', data=df)
plt.title('Credit Score vs Churn')
#plt.show()

#Feature Engineering
features = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])
features["CLV"] = df["Balance"] * df["EstimatedSalary"] / 100000
features["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100], labels=["Young", "MiddleAged", "Senior", "Elderly"])
features["TenureAgeRatio"] = df["Tenure"] / df["Age"]
features = pd.get_dummies(features, columns=['Geography', 'Gender', 'AgeGroup'])
target = df['Exited']

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

#Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

#Model Evaluation and Saving
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} Accuracy: {accuracy}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"--------------------------------")


def evaluate_and_save_model(model, X_train, y_train, X_test, y_test, file_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} Accuracy: {accuracy}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"--------------------------------")

    with open(file_name, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Model saved to {file_name}")
""" 
xgb_model = xgb.XGBClassifier(random_state=42)
#evaluate_and_save_model(xgb_model, X_train, y_train, X_test, y_test, 'xgb_model.pkl')
evaluate_model(xgb_model, X_train, y_train, X_test, y_test)
evaluate_and_save_model(xgb_model, X_resampled, y_resampled, X_test, y_test, 'xgb_model_resampled.pkl')

dt_model = DecisionTreeClassifier(random_state=42)
#evaluate_and_save_model(dt_model, X_train, y_train, X_test, y_test, 'dt_model.pkl')
evaluate_model(dt_model, X_train, y_train, X_test, y_test)

rf_model = RandomForestClassifier(random_state=42)
evaluate_and_save_model(rf_model, X_train, y_train, X_test, y_test, 'rf_model.pkl')

nb_model = GaussianNB()
evaluate_and_save_model(nb_model, X_train, y_train, X_test, y_test, 'nb_model.pkl')

svm_model = SVC(random_state=42)
evaluate_and_save_model(svm_model, X_train, y_train, X_test, y_test, 'svm_model.pkl')

knn_model = KNeighborsClassifier()
evaluate_and_save_model(knn_model, X_train, y_train, X_test, y_test, 'knn_model.pkl')

#Feature Importance
feature_imporance = xgb_model.feature_importances_
feature_names = features.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names, 'Importance': feature_imporance
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

 """
#Voting Classifier
""" 
voting_model = VotingClassifier(
    estimators=[('xgb', xgb.XGBClassifier(random_state=42)), ('rf', RandomForestClassifier(random_state=42)), ('svm', SVC(random_state=42, probability=True))], 
    voting='hard'
)
evaluate_and_save_model(voting_model, X_train, y_train, X_test, y_test, 'voting_model.pkl') """
""" 
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance') """
