from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
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

# 4. Fungsi bantu untuk membuat label 'rendah/sedang/tinggi'
def categorize(item, value):
    low, high = thresholds[item]
    if value <= low:
        return 'rendah'
    elif value <= high:
        return 'sedang'
    else:
        return 'tinggi'

# 5. Tambahkan kolom kelas ke DataFrame berdasarkan kuantil
df['yield_class'] = df.apply(lambda r: categorize(r['Item'], r['hg/ha_yield']), axis=1)

# 6. Siapkan direktori untuk menyimpan model
MODEL_DIR = 'models_per_item'
os.makedirs(MODEL_DIR, exist_ok=True)

# 7. Latih & simpan model per item:
#    - RandomForestRegressor → rf_<item>.pkl
#    - KNeighborsClassifier → knn_<item>.pkl
for item in thresholds.keys():
    # nama file
    safe = item.replace(',', '').replace(' ', '_').lower()
    rf_file  = os.path.join(MODEL_DIR, f"rf_{safe}.pkl")
    knn_file = os.path.join(MODEL_DIR, f"knn_{safe}.pkl")

    # Jika keduanya sudah ada, lewati
    if os.path.exists(rf_file) and os.path.exists(knn_file):
        continue

    # Subset untuk komoditas ini
    df_item = df[df['Item'] == item]
    X = df_item.drop(columns=['hg/ha_yield', 'yield_class', 'Item'])
    y_reg = df_item['hg/ha_yield']
    y_cls = df_item['yield_class']

    # Preprocessor (sama untuk kedua model)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # 7a. Pipeline Random Forest Regressor
    rf_pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    Xr_train, _, yr_train, _ = train_test_split(X, y_reg,
                                                test_size=0.2,
                                                random_state=42)
    rf_pipeline.fit(Xr_train, yr_train)
    joblib.dump(rf_pipeline, rf_file)

    # 7b. Pipeline KNN Classifier
    knn_pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
    Xc_train, _, yc_train, _ = train_test_split(X, y_cls,
                                                test_size=0.2,
                                                random_state=42)
    knn_pipeline.fit(Xc_train, yc_train)
    joblib.dump(knn_pipeline, knn_file)

# 8. Routes Flask
@app.route('/')
def index():
    items = sorted(thresholds.keys())
    areas = sorted(df['Area'].unique().tolist())
    return render_template('home_segmented.html',
                           item_list=items,
                           area_list=areas)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    items = sorted(thresholds.keys())
    areas = sorted(df['Area'].unique().tolist())

    if request.method == 'POST':
        # Ambil input user
        item = request.form['item']
        area = request.form['area']
        year = int(request.form['year'])
        rain = float(request.form['rainfall'])
        pest = float(request.form['pesticides'])
        temp = float(request.form['temp'])

        # Susun DataFrame untuk prediksi
        input_df = pd.DataFrame([{
            'Area': area,
            'Year': year,
            'average_rain_fall_mm_per_year': rain,
            'pesticides_tonnes': pest,
            'avg_temp': temp
        }])

        # Muat dan prediksi nilai kontinu dengan RF
        safe = item.replace(',', '').replace(' ', '_').lower()
        rf_model  = joblib.load(os.path.join(MODEL_DIR, f"rf_{safe}.pkl"))
        rf_pred   = rf_model.predict(input_df)[0]

        # Muat dan prediksi kelas dengan KNN
        knn_model = joblib.load(os.path.join(MODEL_DIR, f"knn_{safe}.pkl"))
        cls_pred  = knn_model.predict(input_df)[0]

        return render_template('result_segmented.html',
                               item=item,
                               rf_result=round(rf_pred, 2),
                               knn_class=cls_pred,
                               area_list=areas,
                               item_list=items)

    # GET → tampilkan form kosong atau hasil default
    return render_template('result_segmented.html',
                           area_list=areas,
                           item_list=items)

@app.route('/referensi')
def referensi():
    return render_template('referensi.html')

if __name__ == '__main__':
    app.run(debug=True)
