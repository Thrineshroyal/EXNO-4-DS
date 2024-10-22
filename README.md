# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

## STEP 1 :

```python
import pandas as pd
data=pd.read_csv("income(1) (1).csv")
data.head()
```

## OUTPUT :

![image](https://github.com/user-attachments/assets/cfec2734-0cd8-4fba-a90a-46bd5784764d)


## STEP 2 :

```python
# Step 2: Clean the Data Set
# Replace '?' with NaN and drop rows with NaN values
data.replace(' ?', pd.NA, inplace=True)
data.dropna(inplace=True)
data.head()
```

## Converting the categorical columns to numerical columns

```python

categorical_cols = ['JobType', 'EdType', 'maritalstatus', 'occupation', 
                    'relationship', 'race', 'gender', 'nativecountry', 'SalStat']
data[categorical_cols] = data[categorical_cols].astype('category')
data.head()

```

## OUTPUT :

![image](https://github.com/user-attachments/assets/7dfe4f83-b776-42d0-a062-65e64afd7a8f)

## STEP 3 :

```python

from sklearn.preprocessing import LabelEncoder, StandardScaler
# Step 3: Apply Feature Scaling
# Convert categorical variables to numeric using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
numeric_features = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

data.head()

```

## OUTPUT :

![image](https://github.com/user-attachments/assets/623c52b8-cfa4-4aca-a7fc-0b0e5f24604b)

## STEP 4 :

```PYTHON

from sklearn.feature_selection import SelectKBest, mutual_info_classif
# Step 4: Apply Feature Selection
X = data.drop(columns=['SalStat'])  # Features
y = data['SalStat']  # Target variable

# Select top k features based on mutual information
k = 5  # Choose number of features to select
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_selected = selector.fit_transform(X, y)

# Get the selected feature names
selected_features = X.columns[selector.get_support()]

# Creating a DataFrame with selected features
selected_data = pd.DataFrame(X_selected, columns=selected_features)

```

## OUTPUT :

![image](https://github.com/user-attachments/assets/0a017bfd-1842-4c2e-b563-95c39a0c3350)

## STEP 5 :

```python

# Step 5: Save the cleaned and processed data to a file
output_file_path = 'processed_income.csv'
selected_data['SalStat'] = y.values  # Append the target variable
selected_data.to_csv(output_file_path, index=False)

print(f"Processed data saved to {output_file_path}")
```

## OUTPUT :

![image](https://github.com/user-attachments/assets/2b4dec41-3170-4efa-9dc8-1fb75a017fad)

# RESULT:

Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
