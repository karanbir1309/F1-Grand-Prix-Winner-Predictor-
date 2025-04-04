# F1-Grand-Prix-Winner-Predictor-
Overview
This project leverages Formula 1 race data to predict driver performance based on lap times, tire compounds, weather conditions, and other metrics. Using the FastF1 library and machine learning techniques, the model analyzes historical race data to forecast driver behavior under specific race conditions.

Features
Data Collection: Fetches detailed race data from Formula 1 sessions using the FastF1 library.

Preprocessing: Cleans and processes lap data, weather data, and driver information for analysis.

Machine Learning:

Implements a Random Forest Regressor to predict driver performance.

Encodes categorical features like drivers and tire compounds using LabelEncoder.

Performance Metrics: Evaluates model accuracy using Mean Absolute Error (MAE).

Simulation: Supports prediction under simulated race conditions.

Installation
Clone the repository:

bash
git clone <repository_url>
cd <repository_name>
Install dependencies:

bash
pip install -r requirements.txt
Key libraries include:

fastf1

pandas

numpy

scikit-learn

Usage
Enable caching for faster data retrieval:

python
fastf1.Cache.enable_cache("fastf1_cache")
Load session data:

python
session = fastf1.get_session(2024, 'Japanese Grand Prix', 'R')
session.load()
Preprocess lap data:

python
lap_data = session.laps[['Driver', 'LapTime', 'Compound', 'TyreLife', 'PitOutTime']]
lap_data['AirTemp'] = session.weather_data['AirTemp']
lap_data['Humidity'] = session.weather_data['Humidity']
lap_data['RainProbability'] = session.weather_data['Rainfall']
Train the model:

python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
Predict driver performance:

python
y_pred = rf_model.predict(X_test)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
Example Prediction
Simulated race conditions:

python
new_race_conditions = pd.DataFrame({
    'LapTime': [85],
    'Compound': [encoded_compound],
    'TyreLife': [10],
    'PitOutTime': [0],
    'AirTemp': [25],
    'Humidity': [60],
    'RainProbability': [20]
})
predicted_driver_name = label_encoder_driver.inverse_transform(predicted_driver_encoded.astype(int))
print("Predicted Driver:", predicted_driver_name[0])
Results
The model achieved a Mean Absolute Error (MAE) of approximately 0.74, indicating good prediction accuracy.

Example prediction output: Predicted Driver: Charles Leclerc

Known Issues
SettingWithCopyWarning: Occurs during DataFrame modifications; can be resolved by using .loc[] indexing.

Limited driver and compound classes may affect prediction accuracy in new scenarios.

Future Enhancements
Expand training data to include multiple seasons for better generalization.

Incorporate advanced feature engineering for improved predictions.

Add support for real-time predictions during live races.

License
This project is licensed under the MIT License.
