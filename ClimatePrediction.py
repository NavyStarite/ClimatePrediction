import requests
import pandas as pd
from joblib import PrintTime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

class PredictData():
    def __init__(self, temp_pred, temp_real, error):
        self.temp_pred = temp_pred
        self.temp_real = temp_real
        self.error = error



#Fetch data from Open-Meteo
print("Fetching historical weather data...")
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":     19.4285,
    "longitude":    -99.1277,
    "start_date":   "2000-11-03",
    "end_date":     "2025-11-03",
    "hourly":       ["temperature_2m", "precipitation"],
    "timezone":      "auto"
}
response = requests.get(url, params=params)
data = response.json()
#print(data)
df = pd.DataFrame({
    "datetime":         pd.to_datetime(data["hourly"]["time"]),
    "temperature":      data["hourly"]["temperature_2m"],
    "precipitation":    data["hourly"]["precipitation"]
})

print("Training model")
#Set date as index
df.set_index("datetime", inplace=True)

#Feature enginering
df["hour"] = df.index.hour
df["day_of_the_year"] = df.index.dayofyear
df["month"] = df.index.month

x = df[["hour", "day_of_the_year", "month", "precipitation"]]
y = df["temperature"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression()
}

results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = mae
    print(f"{name} = Mean Squared Error: {mae:.2f}°C")

#Select Best model
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(f"Best model: {best_model_name}")

predictionData = []


for hourIndex in range(24):
    print(f"Predicting temperature for {hourIndex}:00")
    #Making preditions
    print("Predicting")
    future_date = datetime.datetime.today()
    #print(future_date)
    #print(future_date.timetuple().tm_yday)
    future_features = pd.DataFrame({
        "hour": [hourIndex],
        "day_of_the_year": [future_date.timetuple().tm_yday],
        "month": [future_date.month],
        "precipitation": [0] #Asumiendo que no llueve
    })
    future_prediction = best_model.predict(future_features)
    print(f"Predicted Temperature for: {future_date.strftime('%Y-%m-%d')} using {best_model_name}: {future_prediction[0]:.2f}°C")

    #Real Temp for comparasion
    print("Comparing actual temperature and predicted temperature")

    actual_url = "https://api.open-meteo.com/v1/forecast"
    actual_params = {
        "latitude": 19.4285,
        "longitude": -99.1277,
        "hourly": ["temperature_2m"],
        "timezone": "auto",
        "start_date": future_date.strftime("%Y-%m-%d"),
        "end_date": future_date.strftime("%Y-%m-%d")
    }

    actual_response = requests.get(actual_url, params=actual_params)
    actual_data = actual_response.json()

    if "hourly" in actual_data and "temperature_2m" in actual_data["hourly"]:
        actual_temperature = actual_data["hourly"]["temperature_2m"][hourIndex]
        print(f"Actual Temperature for {future_date.strftime('%Y-%m-%d')}: {actual_temperature:.2f}C")
        error = abs(actual_temperature - future_prediction[0])
        print(f"Prediction Error: {error:.2f}°C")
    else:
        print(f"Actual temperature not available")
        actual_temperature = "NaN"

    predictionData.append(PredictData(future_prediction[0], actual_temperature, error))




for hourIndex in range(24):
    print(f"Predicted: {predictionData[hourIndex].temp_pred}, Actual: {predictionData[hourIndex].temp_real}, Error: {predictionData[hourIndex].error}, Hour: {hourIndex}:00")