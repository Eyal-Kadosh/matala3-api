import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import pickle
from car_data_prep import prepare_data

# קריאת קובץ CSV לתוך DataFrame
file_path = 'C:\\Users\\EyalK\\car_project\\dataset.csv'
df = pd.read_csv(file_path)

# Assuming df is your DataFrame
data = df.copy()

data = prepare_data(data,df)

# Identify non-numeric columns
non_numeric_columns = ['manufactor', 'model', 'Gear', 'Engine_type',"Color"]
# Apply one-hot encoding to non-numeric columns
data= pd.get_dummies(data, columns=non_numeric_columns, drop_first=True)


# מציאת העמודה היעד - מחיר הרכב
target_column = 'Price'

# סינון עמודות שאינן משתמשות לחיזוי
X = data.drop(columns=[target_column])

columns = X.columns
with open('columns_list.pkl', 'wb') as f:
    pickle.dump(columns, f)

# סינון עמודה היעד - מחיר הרכב
y = data[target_column]

# חלוקת הנתונים לסט למידה וסט לבדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# סקלינג של המאפיינים לפי סטנדרטים
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# הגדרת מודל אנד רגרסן עם חוזק אלסטי
model = ElasticNet(alpha=0.1, l1_ratio=0.2)

# התאמה של המודל לנתוני האימון
model.fit(X_train_scaled, y_train)

# חיזוי עם המודל על נתוני הבדיקה
predictions = model.predict(X_test_scaled)

# חישוב שגיאת הריבוע הממוצעת (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# חיזוי תוצאות הסימולציה של סטיית התקן של המודל
std = np.std(predictions)

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_rmse = np.sqrt(-cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error'))

# Grid search for best hyperparameters
param_grid = {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]}
grid_search = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# הדפסת תוצאות החיזוי
print(f"RMSE: {rmse}")
print(f"Model STD: {std}")
print(f"Cross-Validation RMSE: {cv_rmse.mean()} (+/- {cv_rmse.std()})")

# שמירת המודל עם הפארמטרים הטובים ביותר
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# שמירת ה-Scaler כקובץ PKL
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)