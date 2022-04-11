from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from pandas.core.frame import DataFrame
import numpy as np
from functions import model
from functions import data_preprocess


cat_features = [
    'city',
    'type_of_plan',
    'highest_level_education',
    'work_status',
    'sex',
    'relationship_status',
    'reachability',
    'type_of_vehicle']


class User(BaseModel):
    city: Literal[
        'Accra', 'Kumasi', 'Takoradi', 'Cape_coast', 'Tamale']
    type_of_plan: Literal[
        'primary_plan', 'extended_plan', 'premium_plan'
    ]
    highest_level_education: Literal[
        'Bachelor', 'College', 'High School or Below', 'Master', 'Doctor'
    ]
    work_status: Literal[
        'Employed', 'Unemployed', 'Medical Leave', 'Disabled', 'Retired'
    ]
    sex: Literal[
        'female', 'male'
    ]
    relationship_status: Literal[
        'Married', 'Single', 'Divorced'
    ]
    payment_per_month: int
    weeks_since_claim: int
    open_policies: int
    Renew_Offer_Type: int
    reachability: Literal['contacting_agent',
                          'contacting_branch_directly',
                          'customer_call_centre',
                          'via_the_web']
    type_of_vehicle: Literal['4-door_car',
                             '2-door_car',
                             'SUV',
                             'Sports Car',
                             'luxurious_suv',
                             'luxirious_car']


app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Welcome to the churn prediction webpage"}


@app.post("/")
async def inferences(user_data: User):
    ohe = load("outputs/ohe.joblib")
    lb = load("outputs/lb.joblib")
    model_object = load("outputs/model.joblib")
    scaler = load("outputs/scaler.joblib")

    array = np.array([[
                     user_data.city,
                     user_data.type_of_plan,
                     user_data.highest_level_education,
                     user_data.work_status,
                     user_data.sex,
                     user_data.relationship_status,
                     user_data.payment_per_month,
                     user_data.weeks_since_claim,
                     user_data.open_policies,
                     user_data.Renew_Offer_Type,
                     user_data.reachability,
                     user_data.type_of_vehicle,
                     ]])

    df_temp = DataFrame(data=array, columns=[
        'city',
        'type_of_plan',
        'highest_level_education',
        'work_status',
        'sex',
        'relationship_status',
        'payment_per_month',
        'weeks_since_claim',
        'open_policies',
        'Renew_Offer_Type',
        'reachability',
        'type_of_vehicle',
    ])

    X, _, _, _, _ = data_preprocess.process_data(
        df_temp,
        training=False,
        cat_features=model.get_cat_features(),
        lb=lb, ohe=ohe, scaler=scaler)
    prediction = model.inference(model_object, X)
    y = lb.inverse_transform(prediction)[0]
    return {"prediction": y}
