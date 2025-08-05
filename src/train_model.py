import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
import sys

# Load the training data
df = pd.read_csv('data/cleaned_file.csv')

# Drop rows with missing values
original_rows = df.shape[0]
df = df.dropna()
cleaned_rows = df.shape[0]

# If dataset becomes empty, show a message and stop
if cleaned_rows == 0:
    print("❌ All rows had missing values. No data left to train.")
    sys.exit()

print(f"✅ Removed {original_rows - cleaned_rows} rows with missing values. {cleaned_rows} rows left.")

# Split features and target
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model/gaussian_nb_model.pkl')
print("✅ Model saved to model/gaussian_nb_model.pkl")
