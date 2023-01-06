import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from typing import Optional
from unicodedata import category
import json

import bentoml 
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

# from bentoml.yatai.client import get_yatai_client
# custom_url = 'http://es.aidery.io:4000'
# yatai_client = get_yatai_client(custom_url)

class Data(BaseModel):
	Heart: Optional[float] = 0
	Calories: Optional[float] = 0
	Steps: Optional[float] = 0
	Distance: Optional[float] = 0
	Age: Optional[int] = 0
	Gender: str
	Weight: Optional[float] = 0
	# Height: Optional[float] = 0
    
def convertdata(data):
    if data is None:
        data = 0
    else:
        data = data
    return data

async def store_data(data):
    df = pd.DataFrame(columns=['Heart', 'Calories', 'Steps', 'Distance', 'Age', 'Gender', 'Weight'])
    df = df.append({
        'Heart': data.Heart, 'Calories': data.Calories, 'Steps': data.Steps, 'Distance': data.Distance,
        'Age': convertdata(data.Age), 'Gender': data.Gender, 'Weight': convertdata(data.Weight)}, ignore_index=True)

    df_01 = df.copy()
    return df_01

async def covert_obj(data):
    data = data.drop(data[(data['Gender'] == 'other')].index)
    data = data.replace(to_replace=['M', 'F'],value=[0, 1.0])
    return data.copy()
    
# Load model and build service
decision_runner = bentoml.sklearn.get("dt_latest2:latest").to_runner()
svc = bentoml.Service("decision_tree",runners=[decision_runner])

decision_model_loaded = bentoml.sklearn.load_model("dt_latest2:latest")

async def predict_servity(data):
    className = {0: 'Sleep', 1: 'Sedentary', 2: 'Light', 3: 'Moderate', 4: 'Vigorous'}
    X = await store_data(data)
    X = await covert_obj(X)
    pred = decision_model_loaded.predict(X)
    res = pred[0].astype(int)
    catagory = className[res]
    pred_proba = decision_model_loaded.predict_proba(X)
    Max_value_score = pred_proba.argmax(1).item()
    confidence = float(pred_proba[0, Max_value_score])
    return catagory, confidence

# Variable for Input
user_input = JSON(
    pydantic_model=Data,
    validate_json=True,
)

@svc.api(
    input=user_input,
    output=JSON(),
    route='',
)

async def predict_bentoml(data: Data) -> json:
	# input_df = pd.DataFrame([input_data.dict()])
	# return await decision_runner.predict.async_run(input_df)
    catagory, confidence = await predict_servity(data)
    result = {'class': catagory, 'confidence': confidence}
    return {'results': result}



# fastapi_app = FastAPI()
#svc.mount_asgiapp(fastapi_app)
#
#@fastapi_app.get("/metadata")
#def metadata():
#	return {"name":bento_model.tag.name, "version":bento_model.tag.version}
#
## For demo purpose, here's an identical inference endpoint implemented via FastAPI
#@fastapi_app.post("/predict")
#def predict(features: WatchFeatures):
#    input_df = pd.DataFrame([features.dict()])
#    results = decision_runner.predict.run(input_df)
#    return {"prediction": results.tolist()[0]}
#
#
## For demo purpose, here's an identical inference endpoint implemented via FastAPI
#@fastapi_app.post("/predict_fastapi_async")
#async def predict_async(features: WatchFeatures):
#    input_df = pd.DataFrame([features.dict()])
#    results = await decision_runner.predict.async_run(input_df)
#    return {"prediction": results.tolist()[0]}
#
