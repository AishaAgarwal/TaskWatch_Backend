from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

@app.get("/")
def check():
    return {"message":"it is working"}


# This dictionary will store SARIMA models for each user
# user_models = {}

class UserData(BaseModel):
    # user_id: str | None = None
    name: str
    contributionsCollection: dict

def train_sarima_model(df_user):
    p,d,q = 1,1,1
    P,D,Q,s = 1,1,1,2
    df_user['contributions'] = pd.to_numeric(df_user['contributions'], errors='coerce').astype('float64')
    model_user = SARIMAX(df_user['contributions'], order=(p, d, q), seasonal_order=(P, D, Q, s))
    try:
        result_user = model_user.fit()
    
    except Exception as e:
        return None
    
    return result_user
    

def plot_contributions(df_user, result_user, future_dates, predicted_values):
    contributions = df_user.index.tolist()
    contribution_counts = df_user['contributions'].tolist()

    # for week in contributions:
    #     for day in week['contributionDays']:
    #         contribution_dates.append(day['date'])
    #         contribution_counts.append(day['contributionCount'])

    # df_user = pd.DataFrame({'date': pd.to_datetime(contribution_dates), 'contributions': contribution_counts})
    # df_user.set_index('date', inplace=True)

    # Plot the time series with predictions
    plt.figure(figsize=(12, 6))
    plt.plot(contributions,contribution_counts, label='Historical Data')
    plt.plot(future_dates, predicted_values, label='Predicted', color='red')
    user_name = df_user.index.name or 'User'
  
    plt.title(f'GitHub Activity Prediction for User {user_name}')
    plt.xlabel('Date')
    plt.ylabel('Contributions Count')
    plt.legend()

    # Save the plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')

    # Close the plot to avoid memory leaks
    plt.close()

    return img_data

@app.post("/predict")
async def predict(user_data: UserData):
    print("Recieved data: ", user_data.dict())
    # Extract relevant data from the contributionsCollection
    contributions = user_data.contributionsCollection['contributionCalendar']['weeks']
    contribution_dates = []
    contribution_counts = []

    for week in contributions:
        for day in week['contributionDays']:
            contribution_dates.append(day['date'])
            contribution_counts.append(day['contributionCount'])

    df_user = pd.DataFrame({'date': pd.to_datetime(contribution_dates), 'contributions': contribution_counts})
    df_user.set_index('date', inplace=True)

    result_user = train_sarima_model(df_user)
    # user_models[user_data.user_id] = result_user
    

    # Load or create SARIMA model for the user
    # if user_data.user_id not in user_models:
        # Train SARIMA model for the user
    result_user = train_sarima_model(df_user)
    

        # Store the model in the dictionary
        # user_models[user_data.user_id] = result_user
    # else:
    #     # Use the existing model for the user
    #     result_user = user_models[user_data.user_id]

    # Make predictions for the next 7 days
    future_dates = pd.date_range(start=df_user.index[-1], periods=8, freq='D')[1:]
    predictions = result_user.get_forecast(steps=7)

    # Extract predicted values for the next 7 days
    predicted_values = predictions.predicted_mean.round().astype(int).tolist()

    # Generate and return the response
    response = {
        # 'user_id': user_data.user_id,
        'predictions': predicted_values,
        'graph': plot_contributions(df_user, result_user, future_dates, predicted_values)
    }

    return response