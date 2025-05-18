from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import os

app = Flask(__name__)

# Load data
DATA_PATH = 'data/yield_df.csv'
df = pd.read_csv(DATA_PATH).drop(columns=['Unnamed: 0'])

# Konfigurasi fitur
numerical_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
categorical_cols = ['Area']

# Direktori untuk menyimpan model
MODEL_DIR = 'models_per_item'
os.makedirs(MODEL_DIR, exist_ok=True)

# Melatih dan menyimpan model per item jika belum ada
items = df['Item'].unique()
for item in items:
    rf_model_file = os.path.join(MODEL_DIR, f'rf_{item.replace(",", "").replace(" ", "_").lower()}.pkl')
    knn_model_file = os.path.join(MODEL_DIR, f'knn_{item.replace(",", "").replace(" ", "_").lower()}.pkl')

    if os.path.exists(rf_model_file) and os.path.exists(knn_model_file):
        continue

    df_item = df[df['Item'] == item]
    if len(df_item) < 100:
        continue

    X = df_item.drop(columns=['hg/ha_yield', 'Item'])
    y = df_item['hg/ha_yield']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # Random Forest pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # KNN pipeline
    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor(n_neighbors=5))
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_pipeline.fit(X_train, y_train)
    knn_pipeline.fit(X_train, y_train)

    joblib.dump(rf_pipeline, rf_model_file)
    joblib.dump(knn_pipeline, knn_model_file)


@app.route('/')
def index():
    item_list = sorted(df['Item'].unique().tolist())
    area_list = sorted(df['Area'].unique().tolist())
    return render_template('home_segmented.html', item_list=item_list, area_list=area_list)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    item_list = sorted(df['Item'].unique().tolist())
    area_list = sorted(df['Area'].unique().tolist())

    if request.method == 'POST':
        area = request.form['area']
        item = request.form['item']
        year = int(request.form['year'])
        rainfall = float(request.form['rainfall'])
        pesticides = float(request.form['pesticides'])
        temp = float(request.form['temp'])

        input_df = pd.DataFrame([{
            'Area': area,
            'Year': year,
            'average_rain_fall_mm_per_year': rainfall,
            'pesticides_tonnes': pesticides,
            'avg_temp': temp
        }])

        rf_model_file = os.path.join(MODEL_DIR, f'rf_{item.replace(",", "").replace(" ", "_").lower()}.pkl')
        knn_model_file = os.path.join(MODEL_DIR, f'knn_{item.replace(",", "").replace(" ", "_").lower()}.pkl')

        if not os.path.exists(rf_model_file) or not os.path.exists(knn_model_file):
            return f"Model untuk item '{item}' belum tersedia."

        rf_model = joblib.load(rf_model_file)
        knn_model = joblib.load(knn_model_file)

        rf_prediction = rf_model.predict(input_df)[0]
        knn_prediction = knn_model.predict(input_df)[0]

        return render_template('result_segmented.html',
                               rf_result=round(rf_prediction, 2),
                               knn_result=round(knn_prediction, 2),
                               item=item,
                               area_list=area_list,
                               item_list=item_list)

    # Jika GET: tampilkan form kosong
    return render_template('result_segmented.html',
                           area_list=area_list,
                           item_list=item_list)

@app.route('/referensi')
def referensi():
    return render_template('referensi.html')

if __name__ == '__main__':
    app.run(debug=True)
