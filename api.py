from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd
import numpy as np
from car_data_prep import prepare_data

file_path = 'C:\\Users\\EyalK\\car_project\\dataset.csv'
df = pd.read_csv(file_path)

app = Flask(__name__)

# טוען את המודל, הסקיילר ורשימת העמודות
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('columns_list.pkl', 'rb') as f:
    columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # קבלת הנתונים מהבקשה
        data = request.form.to_dict()
      
        for key in data:
             if data[key] == '':
                    data[key] = np.nan

        print("Received data:", data)

        # המרת נתונים מספריים לערכים מספריים אמיתיים
        numeric_fields = ['Year', 'Hand', 'capacity_Engine', 'Pic_num', 'Km','Test' 'Supply_score']
        for field in numeric_fields:
            if field in data:
                try:
                    data[field] = float(data[field])
                except ValueError:
                    data[field] = np.nan

        # המרת נתונים לשדות שאמורים להיות מסוג תאריך
        date_fields = ['Cre_date', 'Repub_date']
        for field in date_fields:
            if field in data:
                try:
                    data[field] = pd.to_datetime(data[field], errors='coerce')
                except ValueError:
                    data[field] = np.nan

        # המרת הנתונים לפורמט DataFrame
        data1= pd.DataFrame([data])
        print("DataFrame after conversion:", data1)

        data1= prepare_data(data1,df)
        print("DataFrame after prepare_data:", data1)


        # בצוע one-hot encoding על הנתונים הנכנסים בהתבסס על רשימת העמודות
        df_encoded = pd.get_dummies(data1, columns=['manufactor', 'model', 'Gear', 'Engine_type', 'Color'])

        # הוספת עמודות חסרות כדי להתאים למודל
        missing_columns = [col for col in columns if col not in df_encoded.columns]
        if missing_columns:
            df_missing = pd.DataFrame(0, index=df_encoded.index, columns=missing_columns)
            df_encoded = pd.concat([df_encoded, df_missing], axis=1)
        df_encoded = df_encoded[columns]

        # סקלינג של הנתונים
        df_scaled = scaler.transform(df_encoded)

        # חיזוי מחיר הרכב
        prediction = model.predict(df_scaled)[0]
        if prediction < 0:
            k= (abs(prediction))
            prediction =( (1/k) *500000)

        # החזרת התוצאה
        return render_template('index.html', prediction=round(prediction, 2), data=request.form)

    except Exception as e:
        print("Error:", e)
        return render_template('index.html', prediction=None, error=str(e), data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
