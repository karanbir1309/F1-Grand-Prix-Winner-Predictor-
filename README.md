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

Key libraries include:
fastf1
pandas
numpy
scikit-learn

Known Issues
SettingWithCopyWarning: Occurs during DataFrame modifications; can be resolved by using .loc[] indexing.
Limited driver and compound classes may affect prediction accuracy in new scenarios.

Future Enhancements
Expand training data to include multiple seasons for better generalization.
Incorporate advanced feature engineering for improved predictions.
Add support for real-time predictions during live races.

License
This project is licensed under the MIT License.
