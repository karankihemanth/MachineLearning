import pandas as pd
df=pd.read_excel("AccidentData .xlsx")
df
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_text
import pandas as pd
# Assuming your dataset is stored in a DataFrame named 'df'
# Features (X) and target variable (y)
X = df.drop('Accident', axis=1)  # Features
y = df['Accident']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display the decision tree rules
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)
# Prompt user for input
gender = int(input("Enter gender (0 for male, 1 for female): "))
road_conditions = int(input("Enter road conditions (0 for bad, 1 for average, 2 for good): "))
weather_conditions = int(input("Enter weather conditions (0 for summer, 1 for foggy, 2 for rainy, 3 for stormy): "))
speed = int(input("Enter speed (numeric value): "))
traffic_density = int(input("Enter traffic density (0 for no traffic, 1 for normal, 2 for high): "))
time_of_day = int(input("Enter time of day (0-23): "))
junction_type = int(input("Enter junction type (0 for no junction, 1 for Y junction, 2 for X junction, 3 for T junction, 4 for O junction, 5 for U junction, 6 for J junction): "))
month = int(input("Enter month (numeric value): "))
population_density = int(input("Enter population density (0 for no population, 1 for normal, 2 for high): "))
alcohol_or_drug_influence = int(input("Enter alcohol or drug influence (0 for no, 1 for yes): "))
# Create a DataFrame with the correct column names and order
# Create a DataFrame with the correct column names and order
user_df = pd.DataFrame({
    'Gender': [gender],  # replace with the user's gender (0 for male, 1 for female)
    'Road Conditions': [road_conditions],  # replace with the user's road conditions
    'Wheather Conditions': [weather_conditions],  # replace with the user's weather conditions
    'speed': [speed],  # replace with the user's speed
    'Traffic density': [traffic_density],  # replace with the user's traffic density
    'Time of Day': [time_of_day],  # replace with the user's time of day
    'Junction Type': [junction_type],  # replace with the user's junction type
    'Month': [month],  # replace with the user's month
    'population density': [population_density],  # replace with the user's population density
    'Alcohol or Drug influence': [alcohol_or_drug_influence]  # replace with the user's alcohol or drug influence
})

# Make a prediction using the trained model (assuming you have a trained model)
prediction = clf.predict(user_df)

# Display the prediction
print("Predicted Accident: ", prediction[0])
