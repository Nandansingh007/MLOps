from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the model
model = joblib.load('E:\MLOps\churnprediction\model\prediction4.joblib')

    
class Customer(BaseModel):
    AccountWeeks: float
    ContractRenewal: int
    DataPlan: int
    DataUsage: float
    CustServCalls: int
    DayMins: float
    DayCalls: int
    MonthlyCharge: float
    OverageFee: float
    RoamMins: float


@app.get("/home")
def home():
    print("hello")
    return {"message": "Hello World"}


#function to print the churn prediction using trained model
@app.post("/infer")
def infer(data:Customer):
    
    # Convert the data to a 2D array
    input_data = [[
        data.AccountWeeks, data.ContractRenewal, data.DataPlan,
        data.DataUsage, data.CustServCalls, data.DayMins,
        data.DayCalls, data.MonthlyCharge, data.OverageFee,
        data.RoamMins
    ]]
    
    # Make predictions using the model
    predictions = model.predict(input_data)
    
    if predictions[0]==1:
        return {"Prediction":"Customer is more likely to churn"}
    else:
        return {"Predictions":"Customer will not churn"}
  
  


if __name__=="__main__":
    uvicorn.run("app:app",host="localhost", port=8084, reload=True)