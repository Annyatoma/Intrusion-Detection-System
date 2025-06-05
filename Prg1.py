import pandas as pd

column_names = [...]  # Define all 42 column names manually or from docs
df = pd.read_csv('KDDTrain+.txt', names=column_names)

from sklearn.preprocessing import LabelEncoder

# Convert categorical columns
encoder = LabelEncoder()
df['protocol_type'] = encoder.fit_transform(df['protocol_type'])
df['service'] = encoder.fit_transform(df['service'])
df['flag'] = encoder.fit_transform(df['flag'])

df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

X = df.drop('label', axis=1)
y = df['label']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

import joblib
joblib.dump(clf, 'ids_dt_model.pkl')

model.save('ids_ann_model.h5')
