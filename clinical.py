# -*- coding: utf-8 -*-

#import libraries
import pandas                as pd
import seaborn               as sns
import matplotlib.pyplot     as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics         import classification_report, confusion_matrix
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import LabelEncoder


# load dataset
col_names = ['case_id',
             'aliquot_quantity',
             'aliquot_volume', 
             'concentration', 
             'age_at_index', 
             'ethnicity', 
             'gender', 
             'race', 
             'vital_status', 
             'year_of_birth', 
             'age_at_diagnosis_adj', 
             'ajcc_pathologic_m', 
             'ajcc_pathologic_n', 
             'ajcc_pathologic_stage', 
             'ajcc_pathologic_t', 
             'ajcc_staging_system_edition', 
             'days_to_last_follow_up', 
             'icd_10_code', 
             'morphology', 
             'primary_diagnosis', 
             'prior_malignancy', 
             'prior_treatment', 
             'progression_or_recurrence', 
             'site_of_resection_or_biopsy', 
             'synchronous_malignancy', 
             'tissue_or_organ_of_origin', 
             'year_of_diagnosis', 
             'treatment_or_therapy', 
             'treatment_type'
             ]
data = pd.read_csv("clinical.csv", header=0, names=col_names)
data["days_to_last_follow_up"] = [float(str(i).replace(",", "")) for i in data["days_to_last_follow_up"]]

#convert data types
new_types = {
    'case_id'                       : 'string',
    'aliquot_quantity'              : 'float',
    'aliquot_volume'                : 'float', 
    'concentration'                 : 'float', 
    'age_at_index'                  : 'int',
    'ethnicity'                     : 'category', 
    'gender'                        : 'category', 
    'race'                          : 'category', 
    'vital_status'                  : 'category', 
    'year_of_birth'                 : 'int',
    'age_at_diagnosis_adj'          : 'float',
    'ajcc_pathologic_m'             : 'category', 
    'ajcc_pathologic_n'             : 'category',  
    'ajcc_pathologic_stage'         : 'category',  
    'ajcc_pathologic_t'             : 'category', 
    'ajcc_staging_system_edition'   : 'category', 
    'days_to_last_follow_up'        : 'int',
    'icd_10_code'                   : 'category', 
    'morphology'                    : 'category', 
    'primary_diagnosis'             : 'category', 
    'prior_malignancy'              : 'category', 
    'prior_treatment'               : 'category',  
    'site_of_resection_or_biopsy'   : 'category',  
    'synchronous_malignancy'        : 'category', 
    'tissue_or_organ_of_origin'     : 'category',  
    'year_of_diagnosis'             : 'int',
    'treatment_or_therapy'          : 'category',  
    'treatment_type'                : 'category'
    }

data = data.astype(new_types)

#encode categorical variables
label_encoder = LabelEncoder()
for column in ['ethnicity', 
               'gender', 
               'race', 
               'vital_status', 
               'ajcc_pathologic_m', 
               'ajcc_pathologic_n',
               'ajcc_pathologic_stage', 
               'ajcc_pathologic_t', 
               'ajcc_staging_system_edition', 
               'icd_10_code', 
               'morphology', 
               'primary_diagnosis', 
               'prior_malignancy', 
               'prior_treatment', 
               'site_of_resection_or_biopsy', 
               'synchronous_malignancy', 
               'tissue_or_organ_of_origin', 
               'treatment_or_therapy',
               'treatment_type'
               ]:
    data[column] = label_encoder.fit_transform(data[column])
    
data = data.dropna().reset_index(drop=True)
print(data['vital_status'])


#split dataset in features and target variable
feature_cols = [
             'aliquot_quantity',
             #'aliquot_volume', 
             'concentration', 
             'gender', 
             'race', 
             'year_of_birth', 
             'age_at_diagnosis_adj', 
             # 'ajcc_pathologic_m', 
             'ajcc_pathologic_n', 
             # 'ajcc_pathologic_stage', 
             # 'ajcc_pathologic_t', 
             # 'ajcc_staging_system_edition', 
             # 'days_to_last_follow_up', 
             # 'icd_10_code', 
             'primary_diagnosis', 
             # 'prior_malignancy', 
             'prior_treatment', 
             'site_of_resection_or_biopsy', 
             # 'synchronous_malignancy', 
             # 'tissue_or_organ_of_origin', 
             # 'year_of_diagnosis', 
             # 'treatment_or_therapy', 
             # 'treatment_type'
             ]
# columns were dropped due to model becoming too accurate and therefore unrealistic

X = data[feature_cols].reset_index(drop=True)  # Feature variables
y = data.vital_status.reset_index(drop=True)   # Target variable

X = X.loc[y.index]
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

#resample: Use oversampling (SMOTE) to balance the classes:

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X, y= smote.fit_resample(X, y)

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a heatmap to check for multicollinearity
feature_names = X.columns
data_df = pd.DataFrame(X, columns=feature_names)
corr_matrix = data_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10}, xticklabels=feature_names, yticklabels=feature_names)
plt.gcf().set_size_inches(20, 16)
plt.title("Feature Correlation Heatmap")
plt.show()

# Build the neural network model
# Add dropout layers to prevent the model from overfitting
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# Add early stopping: stops training when validation performance stops improving:
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


# Evaluate the model
eval_results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {eval_results[0]:.4f}, Test Accuracy: {eval_results[1]:.4f}")

# Predictions and classification report
predictions = (model.predict(X_test) > 0.5).astype(int).flatten()
print("Classification Report:")
print(classification_report(y_test, predictions))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Alive", "Dead"], yticklabels=["Alive", "Dead"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()