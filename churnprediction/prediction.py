import joblib

# Load the model
model = joblib.load('E:\MLOps\churnprediction\model\prediction4.joblib')

#function to print the churn prediction using churn
def prediction(data):
  predictions = model.predict([data])

  if predictions[0] == 1 :
    return "Customer is more likely to churn"
  
  else:
    return "Customer will not churn"

data = [128,1,1,2.70,1,265.1,110,89.0,9.87,10.0]

print(prediction(data))
