from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)

# 1. Load data
DATA_PATH = 'data/yield_df.csv'
df = pd.read_csv(DATA_PATH).drop(columns=['Unnamed: 0'])

# 2. Definisi fitur
numerical_cols   = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
categorical_cols = ['Area']

# 3. Hitung threshold klasifikasi per item (kuantil 33% & 66%)
thresholds = {}
for item in df['Item'].unique():
    df_item = df[df['Item'] == item]
    if len(df_item) < 100:
        continue
    q1 = df_item['hg/ha_yield'].quantile(0.33)
    q2 = df_item['hg/ha_yield'].quantile(0.66)
    thresholds[item] = (q1, q2)

# 4. Direktori model
MODEL_DIR = 'models_per_item'
os.makedirs(MODEL_DIR, exist_ok=True)

# 5. Train & save RF per item (sama seperti sebelumnya)
for item in thresholds.keys():
    rf_file = os.path.join(
    MODEL_DIR,
    f"rf_{item.replace(',', '').replace(' ', '_').lower()}.pkl"
    )   
    if os.path.exists(rf_file):
        continue

    df_item = df[df['Item'] == item]
    X = df_item.drop(columns=['hg/ha_yield','Item'])
    y = df_item['hg/ha_yield']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    rf_pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_pipeline.fit(X_train, y_train)
    joblib.dump(rf_pipeline, rf_file)

# 6. Fungsi bantu klasifikasi
def categorize(item, value):
    """Kembalikan 'rendah','sedang', atau 'tinggi' berdasarkan thresholds[item]."""
    low, high = thresholds[item]
    if value <= low:
        return 'rendah'
    elif value <= high:
        return 'sedang'
    else:
        return 'tinggi'

# 7. Routes
@app.route('/')
def index():
    items = sorted(thresholds.keys())
    areas = sorted(df['Area'].unique().tolist())
    return render_template('home_segmented.html', item_list=items, area_list=areas)

@app.route('/predict', methods=['GET','POST'])
def predict():
    items = sorted(thresholds.keys())
    areas = sorted(df['Area'].unique().tolist())

    if request.method == 'POST':
        # ambil input
        item     = request.form['item']
        area     = request.form['area']
        year     = int(request.form['year'])
        rain     = float(request.form['rainfall'])
        pest     = float(request.form['pesticides'])
        temp     = float(request.form['temp'])

        # buat DataFrame untuk prediksi
        input_df = pd.DataFrame([{
            'Area': area,
            'Year': year,
            'average_rain_fall_mm_per_year': rain,
            'pesticides_tonnes': pest,
            'avg_temp': temp
        }])

        # muat model RF
        rf_file = os.path.join(
            MODEL_DIR,
            f"rf_{item.replace(',', '').replace(' ', '_').lower()}.pkl"
        )
        rf_model = joblib.load(rf_file)

        # prediksi kontinyu
        rf_pred = rf_model.predict(input_df)[0]
        rf_cat  = categorize(item, rf_pred)

        return render_template('result_segmented.html',
                               item=item,
                               rf_result=round(rf_pred,2),
                               rf_class=rf_cat,
                               area_list=areas,
                               item_list=items)

    # GET → form kosong
    return render_template('result_segmented.html',
                           area_list=areas,
                           item_list=items)

@app.route('/referensi')
def referensi():
    return render_template('referensi.html')

if __name__ == '__main__':
    app.run(debug=True)
