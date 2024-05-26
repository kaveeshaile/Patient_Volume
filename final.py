import pickle
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
# Define function to load a model from a pickle file
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Define function to make a prediction using a model
def predict(model, features):
    # Assuming the model expects a DataFrame or NumPy array as input
    # (adjust based on your actual model's requirements)
    return model.predict(features)

# Function to get patient volume prediction for a given model
def get_patient_volume(model_filename, month, year):
    # Load the model
    model = load_model(model_filename)

    # Example features (replace with actual features used by your models)
    features = pd.DataFrame({
        'Month': [month],
        'Year': [year],
        # Add other relevant features for each model
    })

    # Make prediction
    prediction = predict(model, features)
    # Round prediction to integer
    #prediction = int(prediction[0]) 

    return int(round(prediction[0]))  # Assuming the prediction is a single value

# Get user input for month and year
# while True:
#     try:
#         month = int(input("Enter month (1-12): "))
#         if 1 <= month <= 12:
#             break
#         else:
#             print("Invalid month. Please enter a value between 1 and 12.")
#     except ValueError:
#         print("Invalid input. Please enter a number.")

# while True:
#     try:
#         year = int(input("Enter year (YYYY): "))
#         break
#     except ValueError:
#         print("Invalid input. Please enter a number.")

# Define model filenames
model_filenames = {
    'cancer': 'Cancer.pkl',
    'diabetes': 'Diabetes.pkl',
    'obesity': 'Obesity.pkl',
    'arthritis': 'Arthritis.pkl'
}

# # Create a dictionary to store predictions from all models
# predictions = {}
# for model_name, filename in model_filenames.items():
#     prediction = get_patient_volume(filename, month, year)
#     predictions[model_name] = prediction

# # Print predictions
# print("\nPatient Volume Predictions:")
# for model_name, prediction in predictions.items():
#     print(f"{model_name.title()}: {prediction}")


all_predictions = []
# Start date (January 2024)
start_date = datetime(2024, 1, 1)

# End date (December 2025)
end_date = datetime(2025, 12, 31)

# Loop through each month
current_date = start_date
while current_date <= end_date:
    month = current_date.month
    year = current_date.year

    # Get predictions for all models
    predictions = {}
    for model_name, filename in model_filenames.items():
        prediction = get_patient_volume(filename, month, year)
        predictions[model_name] = prediction

    # Print predictions for the current month
    print(f"\nPatient Volume Predictions for {current_date.strftime('%B %Y')}:")
    for model_name, prediction in predictions.items():
        print(f"{model_name.title()}: {prediction}")

    # Increment date to next month
    #current_date += pd.DateOffset(months=1)

    # Create a row for the current month and predictions
    row = {'year': year, 'month': month}
    row.update(predictions)

    # Append the row to the list
    all_predictions.append(row)

    # Increment date to next month
    current_date += pd.DateOffset(months=1)

# Create DataFrame from the prediction list
df = pd.DataFrame(all_predictions)

# Print or save the DataFrame
print(df)

#save csv
df.to_csv('ALL_predictions.csv', index=False)