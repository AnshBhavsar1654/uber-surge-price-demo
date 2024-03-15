import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

historical_data = pd.read_csv('historical_data.csv')
distance_data = pd.read_csv('20citytsp.csv', index_col=0)

def train_regression_model():

    X = historical_data[['Demand']]
    y = historical_data['Fare_Adjustment']

    model = LinearRegression()
    model.fit(X, y)

    return model

def adjust_fare_demand_based(model, demand):
    # Use the trained model to predict fare adjustment based on demand
    fare_adjustment = model.predict([[demand]])
    return fare_adjustment[0]

def generate_histogram_of_demand():
    
    plt.figure(figsize=(8, 6))
    sns.histplot(historical_data['Demand'], kde=True)
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.title('Histogram of Demand')
    plt.show()

def generate_scatter_plot_demand_vs_fare_adjustment(model):
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=historical_data['Demand'], y=historical_data['Fare_Adjustment'])
    plt.xlabel('Demand')
    plt.ylabel('Fare Adjustment')
    plt.title('Scatter Plot of Demand vs. Fare Adjustment')

    demand_values = historical_data['Demand']
    predicted_fare_adjustment = model.predict(np.array(demand_values).reshape(-1, 1))
    plt.plot(demand_values, predicted_fare_adjustment, color='red', linewidth=2)
    plt.legend(['Regression Line'])

    plt.show()

def main():

    start_location = int(input("Enter the start destination (1-20): "))
    end_location = int(input("Enter the end destination (1-20): "))

    if start_location < 1 or start_location > 20 or end_location < 1 or end_location > 20:
        print("Invalid input. Start and end destinations should be between 1 and 20.")
        return

    # Calculate the distance between the start and end destinations
    distance = distance_data.iloc[start_location - 1, end_location - 1]

    # Generate a random demand value between 0.7 and 4.5
    demand = round(random.uniform(0.7, 4.5), 1)

    # Base fare and fare per kilometer
    base_fare = 100
    fare_per_km_normal_demand = 12

    # Use the trained model to adjust the fare per kilometer based on demand
    model = train_regression_model()
    fare_adjustment = adjust_fare_demand_based(model, demand)
    fare_adjustment = round(fare_adjustment, 2)
    
    total_fare=base_fare+(fare_per_km_normal_demand + fare_adjustment)*distance
    total_fare=round(total_fare, 2)
    
    print("Distance: ", distance)
    print("Demand: ", demand)
    print("Fare Adjustment: ", fare_adjustment)
    print("Total Fare: ", total_fare)

    # Generate the three specified graphs
    generate_histogram_of_demand()
    generate_scatter_plot_demand_vs_fare_adjustment(model)

if __name__ == '__main__':
    main()
