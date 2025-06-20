# 1. Install Required Libraries (Only needed once)
!pip install opendatasets --quiet

# 2. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
print("Libraries imported successfully!")

# 3. Mount Google Drive (only for Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# 4. Load Dataset from Google Drive
try:
    # Define the correct path to your CSV file
    data_path = "/content/drive/MyDrive/Projects/Task2/train.csv"

    # Load the dataset
    titanic = pd.read_csv(data_path)
    print("✅ Loaded dataset from Google Drive:", data_path)

    # Standardize column names if necessary
    titanic.columns = titanic.columns.str.lower()
    titanic = titanic.rename(columns={
        'passengerid': 'passenger_id',
        'survived': 'survived',
        'pclass': 'pclass',
        'name': 'name',
        'sex': 'sex',
        'age': 'age',
        'sibsp': 'sibsp',
        'parch': 'parch',
        'ticket': 'ticket',
        'fare': 'fare',
        'cabin': 'cabin',
        'embarked': 'embarked'
    })

except FileNotFoundError:
    print("❌ Dataset not found in Google Drive. Falling back to seaborn dataset...")
    titanic = sns.load_dataset('titanic')
    print("✅ Loaded seaborn dataset as fallback")

    # Rename columns to match expected format
    titanic = titanic.rename(columns={
        'survived': 'survived',
        'pclass': 'pclass',
        'sibsp': 'sibsp',
        'parch': 'parch'
    })

print(f"📊 Dataset shape: {titanic.shape}")

# 5. Initial Data Exploration
print("\n" + "="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"\nDataset Shape: {titanic.shape}")
print(f"Number of rows: {titanic.shape[0]}")
print(f"Number of columns: {titanic.shape[1]}")
print("\nColumn Names and Data Types:")
print(titanic.dtypes)
print("\nFirst 5 rows:")
print(titanic.head())
print("\nBasic Statistical Summary:")
print(titanic.describe())

# 6. Data Quality Assessment
print("\n" + "="*50)
print("DATA QUALITY ASSESSMENT")
print("="*50)

# Check for missing values
print("\nMissing Values Count:")
missing_data = titanic.isnull().sum()
missing_percentage = (missing_data / len(titanic)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Percentage': missing_percentage
})
print(missing_df[missing_df['Missing Count'] > 0])

# Check for duplicates
print(f"\nDuplicate rows: {titanic.duplicated().sum()}")

# Check unique values in categorical columns
print("\nUnique values in categorical columns:")
categorical_cols = titanic.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {titanic[col].unique()}")

# 7. Data Cleaning
print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Create a copy for cleaning
titanic_clean = titanic.copy()

# Handle missing values in 'age'
print(f"Age missing values before cleaning: {titanic_clean['age'].isnull().sum()}")
# Fill missing ages with median age by passenger class and gender
titanic_clean['age'] = titanic_clean.groupby(['pclass', 'sex'])['age'].transform(
    lambda x: x.fillna(x.median())
)
print(f"Age missing values after cleaning: {titanic_clean['age'].isnull().sum()}")

# Handle missing values in 'embarked'
print(f"Embarked missing values before cleaning: {titanic_clean['embarked'].isnull().sum()}")
# Fill missing embarked with mode (most common port)
if titanic_clean['embarked'].isnull().sum() > 0:
    most_common_port = titanic_clean['embarked'].mode()[0]
    titanic_clean['embarked'].fillna(most_common_port, inplace=True)
    print(f"Most common embarked port: {most_common_port}")
print(f"Embarked missing values after cleaning: {titanic_clean['embarked'].isnull().sum()}")

# Handle missing values in 'cabin' (if present)
if 'cabin' in titanic_clean.columns:
    print(f"Cabin missing values: {titanic_clean['cabin'].isnull().sum()}")
    titanic_clean['cabin'].fillna('Unknown', inplace=True)

# Handle missing values in 'deck' (if present)
if 'deck' in titanic_clean.columns:
    print(f"Deck missing values: {titanic_clean['deck'].isnull().sum()}")
    titanic_clean['deck'].fillna('Unknown', inplace=True)

# Create age groups for better analysis
titanic_clean['age_group'] = pd.cut(titanic_clean['age'],
                                   bins=[0, 12, 18, 35, 60, 100],
                                   labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# Create fare groups
titanic_clean['fare_group'] = pd.cut(titanic_clean['fare'],
                                    bins=[0, 10, 30, 100, 1000],
                                    labels=['Low', 'Medium', 'High', 'Very High'])

# Create family size feature
titanic_clean['family_size'] = titanic_clean['sibsp'] + titanic_clean['parch'] + 1
print("\nData cleaning completed!")
print(f"Final dataset shape: {titanic_clean.shape}")

# 8. Exploratory Data Analysis - Overall Survival
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Overall survival statistics
survival_counts = titanic_clean['survived'].value_counts()
survival_rate = titanic_clean['survived'].mean()
print(f"Overall Survival Rate: {survival_rate:.1%}")
print(f"Survivors: {survival_counts[1]}")
print(f"Non-survivors: {survival_counts[0]}")

# Visualization 1: Overall Survival Overview
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Titanic Survival Analysis Overview', fontsize=16, fontweight='bold')

# Overall survival rate pie chart
axes[0,0].pie(survival_counts.values, labels=['Died', 'Survived'], autopct='%1.1f%%',
              colors=['#ff6b6b', '#4ecdc4'])
axes[0,0].set_title(f'Overall Survival Rate: {survival_rate:.1%}')

# Survival by gender
survival_by_gender = pd.crosstab(titanic_clean['sex'], titanic_clean['survived'], normalize='index')
survival_by_gender.plot(kind='bar', ax=axes[0,1], color=['#ff6b6b', '#4ecdc4'])
axes[0,1].set_title('Survival Rate by Gender')
axes[0,1].set_xlabel('Gender')
axes[0,1].set_ylabel('Survival Rate')
axes[0,1].legend(['Died', 'Survived'])
axes[0,1].tick_params(axis='x', rotation=0)

# Survival by passenger class
survival_by_class = pd.crosstab(titanic_clean['pclass'], titanic_clean['survived'], normalize='index')
survival_by_class.plot(kind='bar', ax=axes[1,0], color=['#ff6b6b', '#4ecdc4'])
axes[1,0].set_title('Survival Rate by Passenger Class')
axes[1,0].set_xlabel('Passenger Class')
axes[1,0].set_ylabel('Survival Rate')
axes[1,0].legend(['Died', 'Survived'])
axes[1,0].tick_params(axis='x', rotation=0)

# Survival by age group
survival_by_age = pd.crosstab(titanic_clean['age_group'], titanic_clean['survived'], normalize='index')
survival_by_age.plot(kind='bar', ax=axes[1,1], color=['#ff6b6b', '#4ecdc4'])
axes[1,1].set_title('Survival Rate by Age Group')
axes[1,1].set_xlabel('Age Group')
axes[1,1].set_ylabel('Survival Rate')
axes[1,1].legend(['Died', 'Survived'])
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Additional sections remain unchanged — continue with full EDA and visualization...

# [You can paste the rest of the original EDA and visualization code here]
