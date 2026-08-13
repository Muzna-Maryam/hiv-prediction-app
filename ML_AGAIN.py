"""
Project Title: AIDS Outcome Prediction using Machine Learning
Description: This script develops and evaluates multiple machine learning models to predict AIDS treatment outcomes based
             on clinical data. The dataset includes various features such as demographic information, medical history,
             treatment details, and laboratory results. The models explored include Random Forest, Gradient Boosting,
             Naive Bayes, K-Nearest Neighbors, Logistic Regression, Support Vector Machines, XGBoost, and an Artificial
             Neural Network using TensorFlow.

Dataset: AIDS_CLASSIFICATION.csv

Functions Included:
- handle_outliers(df, column): Corrects outliers in the specified dataframe column
- Cls_model_GrdSrch_Tune(model, X, y, params): Performs GridSearchCV to tune model parameters
- Additional plotting and data transformation functions

Output: Trains and evaluates models, displaying accuracy metrics and predictions. Saves the best model parameters.

AZBABANU ENGINEER
MUZNA MARYAM
"""
###IMPORTING LIBRARIES#####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import seaborn as sns

##### Loads the data####
file_path = r"D:\Fourth Semester\Machine Learning\datasets\AIDS_CLASSIFICATION.csv.csv"
data = pd.read_csv(file_path)

###DATA PREPROCESSING#####
print(data.head())
print(data.describe().T)

print(data.info())
data.drop(columns=['race'], inplace=True)

##### Define the continuous columns and  visualize em######
continuous_columns = ['trt', 'age', 'wtkg', 'karnof', 'cd40', 'cd420', 'cd80', 'cd820', 'strat', 'preanti']
colors = ['#1f77b4', '#aec7e8', '#ff7f0e']

# Create subplots
fig, axes = plt.subplots(len(continuous_columns), 1, figsize=(8, 24))

# Plot KDE plots for each column
for i, col in enumerate(continuous_columns):
    color = colors[i % len(colors)]
    sns.kdeplot(data[col], color=color, ax=axes[i])
    axes[i].set_title(f'KDE Plot of {col}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.show()

infected_counts = data['infected'].value_counts()

# Define labels for the pie chart
labels = ['Not Infected', 'Infected']

# Define new colors for the pie chart to enhance visibility
new_colors = ['#4e79a7', '#f28e2b']  # Blue for not infected, Orange for infected

# Create the pie chart with black outlines
fig, ax = plt.subplots()
ax.pie(infected_counts, labels=labels, colors=new_colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
ax.set_title('Pie Chart of Infection Status ')
plt.show()

# Feature Engineering/ Scaling/ Lifting
data2 = data.copy()

# Check for zero or negative values in 'preanti' column
if (data2['preanti'] <= 0).any():
    data2 = data2[data2['preanti'] > 0]

# Apply log transformation to 'preanti' column
data2['log_preanti'] = np.log(data2['preanti'])

# Create the overall health score
health_columns = ['karnof', 'cd40', 'cd420', 'cd80', 'cd820']
data2['healthscore'] = data2[health_columns].mean(axis=1)

# Squaring selected features to capture non-linear relationships
features_to_square = ['age', 'wtkg', 'cd40', 'cd420', 'cd80', 'cd820']
for feature in features_to_square:
    squared_feature = feature + '_squared'
    data2[squared_feature] = data2[feature] ** 2

data2.describe()

# Additional Visualizations
# Bar plot for treatment groups vs infection status
trt_infected_counts = data.groupby(['trt', 'infected']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
trt_infected_counts.plot(kind='bar', stacked=True, color=['#4e79a7', '#f28e2b'], ax=ax)
ax.set_title('Distribution of Infection Status by trt')
ax.set_xlabel('Treatment Group')
ax.set_ylabel('Count')
ax.legend(['Not Infected', 'Infected'], title='Infection Status')
plt.show()

# Bar plot for Infected and Non-Infected Individuals by Treatment Used
# Map treat values to desired labels
data.loc[:, 'treat_label'] = data['treat'].map({0: 'ZDV', 1: 'Others'})

# Bar plot using seaborn with hue and custom color palette
plt.figure(figsize=(8, 6))
sns.countplot(x='treat_label', hue='infected', data=data,
              palette=['#0E4C92', '#BE5504'],
              order=['ZDV', 'Others'])

# Adjust y-axis ticks based on the count of cases
max_count = data['treat_label'].value_counts().max()
plt.yticks(range(0, max_count + 1, 100))  # Adjust the step size as needed
plt.ylim(0, max_count + 100)  # Add a buffer for better visualization

plt.title('Count of Infected and Non-Infected Individuals by Treatment Used')
plt.xlabel('Treatment Used')
plt.ylabel('Count')
plt.legend(title='Infected', loc='upper left', labels=['Non-Infected', 'Infected'])
plt.show()

# Visualization: Bar Graph for Health Score vs. Infection Status
plt.figure(figsize=(10, 6))
sns.barplot(x='infected', y='healthscore', data=data2, hue='infected', dodge=False, palette=['#003151', '#8D4004'])
plt.title('Health Score vs Infection Status')
plt.xlabel('Infection Status')
plt.ylabel('Health Score')
plt.xticks([0, 1], ['Not Infected', 'Infected'])
plt.legend().remove()
plt.show()

# Separate features and target
X = data2.drop(columns=['infected'])
y = data2['infected']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform feature selection
k_best = SelectKBest(score_func=f_classif, k='all')
X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
X_test_selected = k_best.transform(X_test_scaled)

# Define consolidated models and parameter grids
models_param_grids = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'LDA': {
        'model': LinearDiscriminantAnalysis(),
        'param_grid': {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': ['auto', 0.1, 0.5, 1],
            'tol': [0.0001, 0.001, 0.01, 0.1]
        }
    },
    'GaussianNB': {
        'model': GaussianNB(),
        'param_grid': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'param_grid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }
}


results = {}

##### Hyperparameter Tuning(Optimization) and Model Evaluation && Cross-validation#####
for model_name, model_info in models_param_grids.items():
    model = model_info['model']
    param_grid = model_info['param_grid']

    print(f"Hyperparameter tuning for {model_name}...")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    if model_name == 'SVM':
        y_score = best_model.decision_function(X_test_scaled)
        roc_auc = roc_auc_score(y_test, y_score)

        # Calculate ROC curve and ROC area for SVM
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for SVM')
        plt.legend(loc="lower right")
        plt.show()
    else:
        roc_auc = None

    # Store results
    results[model_name] = {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'roc_auc': roc_auc
    }

    print(f"Accuracy for {model_name}: {accuracy}")
    print(f"Classification Report for {model_name}:")
    print(class_report)
    print("\n" + "-"*60 + "\n")

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Infected', 'Infected'],
                yticklabels=['Not Infected', 'Infected'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot feature importances for RandomForestClassifier
    if model_name == 'RandomForest':
        importances = best_model.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title('Feature Importances from RandomForestClassifier')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()