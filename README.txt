single unified Model
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

# ——— Load & persiapan data ——————————————
df = pd.read_csv('data/yield_df.csv').drop(columns=['Unnamed: 0'])
X = df.drop(columns=['hg/ha_yield'])
y = df['hg/ha_yield']

categorical_cols = ['Area', 'Item']
numerical_cols   = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)

knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=5))
])
knn_pipeline.fit(X_train, y_train)


# ——— Route untuk landing page ————————————
@app.route('/')
def index():
    return render_template('index.html')


# ——— Route untuk form prediksi ———————————
@app.route('/home', methods=['GET', 'POST'])
def home():
    area_list = sorted(df['Area'].unique().tolist())
    item_list = sorted(df['Item'].unique().tolist())

    rf_result  = None
    knn_result = None

    if request.method == 'POST':
        # ambil input
        area       = request.form['area']
        item       = request.form['item']
        year       = int(request.form['year'])
        rainfall   = float(request.form['rainfall'])
        pesticides = float(request.form['pesticides'])
        temp       = float(request.form['temp'])

        # DataFrame untuk prediksi
        input_df = pd.DataFrame([{
            'Area': area,
            'Item': item,
            'Year': year,
            'average_rain_fall_mm_per_year': rainfall,
            'pesticides_tonnes': pesticides,
            'avg_temp': temp
        }])

        # prediksi
        rf_result  = rf_pipeline.predict(input_df)[0]
        knn_result = knn_pipeline.predict(input_df)[0]

    return render_template('home.html',
        area_list=area_list,
        item_list=item_list,
        rf_result=rf_result,
        knn_result=knn_result
    )


if __name__ == '__main__':
    app.run(debug=True)
