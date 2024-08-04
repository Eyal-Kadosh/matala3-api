import pandas as pd
import numpy as np
import re
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

##   *הערה חשובה*
## על מנת שפונקציית הכנת הנתונים תעבוד גם על הנתונים המתקבלים מאתר האפליקציה שעליהם צריך לבצע את החיזוי, ביצעתי מספר שינויים הכרחיים 
## השינוי המרכזי הוא שכעת הפונקציה מקבלת 2 ארגומנטיים מסוג בסיסי נתונים וממלאת את הערכים החסרים בחדשה לפי הערכים שכבר כתבנו בעבר מהדאטה המקורית


def prepare_data(data, df):
    # טיפול בערכים חסרים - לדוגמא, נמלא ערכים חסרים בממוצע העמודה

    ##### GEAR
    most_common_Gear = df['Gear'].mode()[0]
    data['Gear'] = data['Gear'].fillna(most_common_Gear)

    ##### CAPACITY_ENGINE
    if data['capacity_Engine'].dtype == object:
        data['capacity_Engine'] = data['capacity_Engine'].astype(str).str.replace(',', '')
    if df['capacity_Engine'].dtype == object:
        df['capacity_Engine'] = df['capacity_Engine'].astype(str).str.replace(',', '')
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
    data['capacity_Engine'] = pd.to_numeric(data['capacity_Engine'], errors='coerce')
    median_values = df.groupby('model')['capacity_Engine'].transform('median')
    data['capacity_Engine'] = data['capacity_Engine'].fillna(median_values)
    median_capacity_engine_tot = df['capacity_Engine'].median()
    data['capacity_Engine'] = data['capacity_Engine'].fillna(median_capacity_engine_tot)

    Q1 = df['capacity_Engine'].quantile(0.25)
    Q3 = df['capacity_Engine'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['capacity_Engine'] >= lower_bound) & (data['capacity_Engine'] <= upper_bound)]

    ##### ENGINE_TYPE
    most_common_eng_type = df['Engine_type'].mode()[0]
    data['Engine_type'] = data['Engine_type'].fillna(most_common_eng_type)

    ##### PREV/CURR_OWNERSHIP
    most_common_owner = df['Prev_ownership'].mode()[0]
    data['Prev_ownership'] = data['Prev_ownership'].fillna(most_common_owner)
    data['Curr_ownership'] = data['Curr_ownership'].fillna(most_common_owner)

    ###### PIC_NUM
    median_pic_num = df['Pic_num'].median()
    data['Pic_num'] = data['Pic_num'].fillna(median_pic_num)

    ###### COLOR
    data['Color'] = data['Color'].fillna('Unknown')
    
    ###### AREA
    data['Area'] = data['Area'].fillna('Unknown')
    
    ###### CITY
    data['City'] = data['City'].fillna('Unknown')

    ###### KM
    if data['Km'].dtype == object:
        data['Km'] = data['Km'].astype(str).str.replace(',', '')
    if df['Km'].dtype == object:
        df['Km'] = df['Km'].astype(str).str.replace(',', '')
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    data['Km'] = pd.to_numeric(data['Km'], errors='coerce')
    df.loc[df['Km'] < 999, 'Km'] *= 1000
    df['Km'] = df['Km'].replace(0, np.nan)
    data.loc[data['Km'] < 999, 'Km'] *= 1000
    data['Km'] = data['Km'].replace(0, np.nan)
    
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Year']
    data['Age'] = current_year - data['Year']
    valid_km_data = df.dropna(subset=['Km'])
    average_kms_per_year = valid_km_data['Km'].sum() / valid_km_data['Age'].sum()

    def fill_missing_km(row):
        if pd.isna(row['Km']):
            estimated_kms = row['Age'] * average_kms_per_year
            return estimated_kms
        else:
            return row['Km']
    data['Km'] = data.apply(fill_missing_km, axis=1)

    Q1 = df['Km'].quantile(0.25)
    Q3 = df['Km'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['Km'] <= upper_bound)]
    
    df = df.drop(columns=['Age'])
    data = data.drop(columns=['Age'])
   
    ###### MANUFACTOR
    data['manufactor'] = data['manufactor'].replace('Lexsus', 'לקסוס')

    ###### HAND
    median_Hand = df['Hand'].median()
    data['Hand'] = data['Hand'].fillna(median_Hand)

    ##### MODEL
    def remove_years(value):
        return re.sub(r'\s*\(\d{4}\)', '', value)  
    data['model'] = data['model'].apply(remove_years)

    ##### DROPED COLUMNS
    data.drop(columns=["Description", "City", "Cre_date", "Repub_date", "Test", "Supply_score", "Prev_ownership", "Curr_ownership", "Area"], inplace=True)

    return data


