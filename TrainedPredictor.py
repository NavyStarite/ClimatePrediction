import pickle
import datetime
import pandas as pd
import requests

class PredictData():
    def __init__(self, temp_pred, temp_real, error):
        self.temp_pred = temp_pred
        self.temp_real = temp_real
        self.error = error

filename = 'best_trained_model.pkl'
model = pickle.load(open(filename, 'rb'))
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
    future_prediction = model.predict(future_features)
    print(f"Predicted Temperature for: {future_date.strftime('%Y-%m-%d')}: {future_prediction[0]:.2f}°C")

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
        error = 0
        actual_temperature = 0

    predictionData.append(PredictData(future_prediction[0], actual_temperature, error))




for hourIndex in range(24):
    print(f"Predicted: {predictionData[hourIndex].temp_pred}, Actual: {predictionData[hourIndex].temp_real}, Error: {predictionData[hourIndex].error}, Hour: {hourIndex}:00")

