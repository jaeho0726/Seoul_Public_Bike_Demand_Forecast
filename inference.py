import joblib
import pandas as pd

preprocessor = joblib.load("models/preprocessor.pkl")

use_count_model = joblib.load("models/use_count_model.pkl")

avg_use_time_model = joblib.load("models/avg_use_time_model.pkl")

model_info = joblib.load("models/model_info.pkl")