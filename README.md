
# Temperature Conversion Model

This project implements a linear regression model to convert temperatures from Celsius to Fahrenheit. The model is trained using synthetic data, and the performance of the model is evaluated over time. 


## Features

- **Linear Regression Model**: Trains a linear regression model to convert Celsius to Fahrenheit.
- **Model Evaluation**: Plots the Mean Squared Error (MSE) of the model over time as the number of data points increases.
- **Continuous User Input**: Allows users to continuously input Celsius values and get the corresponding Fahrenheit conversion until a non-numeric value is entered.

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_temperature_conversion_model():
    """
    Train a linear regression model to convert Celsius to Fahrenheit and plot the model accuracy over time.

    Returns:
    - model: The trained linear regression model.
    """
    # Step 1: Generate Data
    celsius = np.linspace(-100, 100, 200)  # Generate 200 data points from -100 to 100 degrees Celsius
    fahrenheit = (celsius * 9/5) + 32  # Convert to Fahrenheit

    # Prepare the data for training
    X = celsius.reshape(-1, 1)  # Features (Celsius)
    y = fahrenheit  # Target (Fahrenheit)

    # Step 2: Train the Model and Step 3: Evaluate the Model
    # We'll simulate training over time by incrementally increasing the data size
    mse_values = []
    data_sizes = range(10, 201, 10)  # Incrementally increasing data sizes

    for size in data_sizes:
        # Use only the first `size` data points
        X_subset = X[:size]
        y_subset = y[:size]

        # Train the model
        model = LinearRegression()
        model.fit(X_subset, y_subset)

        # Predict and evaluate
        y_pred = model.predict(X_subset)
        mse = mean_squared_error(y_subset, y_pred)
        mse_values.append(mse)

    # Step 4: Plot Model Accuracy Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, mse_values, marker='o')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

    return model

def convert_celsius_to_fahrenheit(model, celsius_value):
    """
    Use the trained model to convert Celsius to Fahrenheit.

    Parameters:
    - model: The trained linear regression model.
    - celsius_value: Temperature in Celsius to be converted.

    Returns:
    - Temperature in Fahrenheit.
    """
    celsius_array = np.array([[celsius_value]])
    fahrenheit_value = model.predict(celsius_array)
    return fahrenheit_value[0]

# Train the model and plot the accuracy
model = train_temperature_conversion_model()

# Continuous input loop for conversion
while True:
    celsius_value = input("Enter temperature value in Celsius to convert to Fahrenheit (or type any letter to exit): ")
    if celsius_value.isalpha():
        print("Exiting the program.")
        break
    try:
        celsius_value = float(celsius_value)
        fahrenheit_value = convert_celsius_to_fahrenheit(model, celsius_value)
        print("*" * 20)
        print(f"{celsius_value} degrees Celsius is {fahrenheit_value:.2f} degrees Fahrenheit.")
        print("*" * 20)
    except ValueError:
        print("Please enter a valid numeric value for Celsius.")
```

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/temperature-conversion-model.git
   cd temperature-conversion-model
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

3. **Run the Script**:
   ```bash
   python temperature_conversion.py
   ```

4. **Use the Continuous Input Loop**:
   - Enter temperature values in Celsius to get the Fahrenheit conversion.
   - To exit, type any alphabetic character.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
