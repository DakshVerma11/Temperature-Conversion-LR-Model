
# Temperature Conversion Model

This project implements a linear regression model to convert temperatures from Celsius to Fahrenheit. The model is trained using synthetic data, and the performance of the model is evaluated over time. 


## Features

- **Linear Regression Model**: Trains a linear regression model to convert Celsius to Fahrenheit.
- **Model Evaluation**: Plots the Mean Squared Error (MSE) of the model over time as the number of data points increases.
- **Continuous User Input**: Allows users to continuously input Celsius values and get the corresponding Fahrenheit conversion until a non-numeric value is entered.



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
