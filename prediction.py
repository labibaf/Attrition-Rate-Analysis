import pandas as pd
import joblib

# Load model 
best_model = joblib.load('./model/best_model.pkl')

# Load data untuk prediksi
data_to_predict = pd.read_csv('data_to_predict.csv') 

# Melakukan prediksi
predictions = best_model.predict(data_to_predict) 

result_df = pd.DataFrame({'Prediction': predictions})
result_df.to_csv('prediction_result.csv', index=False)

print("Hasil prediksi telah disimpan ke dalam file 'prediction_result.csv'")
