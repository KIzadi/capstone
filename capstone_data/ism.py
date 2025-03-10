import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class NeuralNetworkFromScratch:
    """
    A fully-connected neural network implemented from scratch for volatility prediction.
    """
    
    def __init__(self, layer_sizes, activation='tanh', learning_rate=0.01, l2_reg=0.001):
        """
        Initialize the neural network.
        
        Parameters:
        -----------
        layer_sizes : list
            List containing the number of neurons in each layer including input and output layers
        activation : str
            Activation function to use ('tanh', 'relu', or 'sigmoid')
        learning_rate : float
            Learning rate for gradient descent
        l2_reg : float
            L2 regularization parameter
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        
        # Set activation function
        if activation == 'tanh':
            self.activation = self._tanh
            self.activation_derivative = self._tanh_derivative
        elif activation == 'relu':
            self.activation = self._relu
            self.activation_derivative = self._relu_derivative
        elif activation == 'sigmoid':
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        else:
            raise ValueError("Unsupported activation function")
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        self.loss_history = []
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        activations : list
            List of activations for each layer
        Z_values : list
            List of weighted inputs for each layer
        """
        activations = [X]
        Z_values = []
        
        for i in range(len(self.weights)):
            Z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            Z_values.append(Z)
            if i == len(self.weights) - 1:
                A = Z 
            else:
                A = self.activation(Z)
                
            activations.append(A)
            
        return activations, Z_values
    
    def backward(self, X, y, activations, Z_values):
        """
        Backward propagation to update weights and biases.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)
        y : numpy.ndarray
            Target values of shape (n_samples, n_outputs)
        activations : list
            List of activations from forward propagation
        Z_values : list
            List of weighted inputs from forward propagation
            
        Returns:
        --------
        gradients_w : list
            List of gradients for weights
        gradients_b : list
            List of gradients for biases
        """
        m = X.shape[0]
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (for MSE loss)
        delta = activations[-1] - y
        
        for l in range(len(self.weights) - 1, -1, -1):
            gradients_w[l] = (1/m) * np.dot(activations[l].T, delta) + (self.l2_reg * self.weights[l])
            gradients_b[l] = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.activation_derivative(Z_values[l-1])
                
        return gradients_w, gradients_b
    
    def train(self, X, y, epochs=1000, batch_size=32, validation_data=None, patience=50, verbose=True):
    
        n_samples = X.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                activations, Z_values = self.forward(X_batch)
                
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, Z_values)
                
                for l in range(len(self.weights)):
                    self.weights[l] -= self.learning_rate * gradients_w[l]
                    self.biases[l] -= self.learning_rate * gradients_b[l]
            
            activations, _ = self.forward(X)
            y_pred = activations[-1]
            train_loss = np.mean((y_pred - y)**2)
            self.loss_history.append(train_loss)
            
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_activations, _ = self.forward(X_val)
                y_val_pred = val_activations[-1]
                val_loss = np.mean((y_val_pred - y_val)**2)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                val_msg = f", Validation Loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.6f}{val_msg}")
                
        return self.loss_history
    
    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]


def load_and_preprocess_spy_data(file_path="spy_stock_data.csv"):

    try:
        df = pd.read_csv(file_path, skiprows=2)
        
        column_mapping = {
            df.columns[0]: "Date",
            df.columns[1]: "Close",
            df.columns[2]: "High", 
            df.columns[3]: "Low",
            df.columns[4]: "Open",
            df.columns[5]: "Volume"
        }
        df = df.rename(columns=column_mapping)
        
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Data loaded successfully: {df.shape[0]} rows from {df.index.min()} to {df.index.max()}")
        
        df['Returns'] = df['Close'].pct_change()
        
        df['Volatility_5d'] = df['Returns'].rolling(window=5).std() * np.sqrt(252)
        df['Volatility_21d'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)  # ~1 month
        df['Volatility_63d'] = df['Returns'].rolling(window=63).std() * np.sqrt(252)  # ~3 months
        
        df['Log_Price'] = np.log(df['Close'])
        df['Price_Change'] = df['Close'].pct_change(5)
        
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        
        # True Range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR_14'] = df['TR'].rolling(window=14).mean()
        
        # Volume-based features
        df['Volume_Change'] = df['Volume'].pct_change(5)
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
        
        # Return-based features
        df['Abs_Returns'] = np.abs(df['Returns'])
        df['Returns_MA5'] = df['Returns'].rolling(window=5).mean()
        df['Returns_Std5'] = df['Returns'].rolling(window=5).std()
        
        # RSI 
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df.dropna(inplace=True)
        
        print(f"Preprocessed data shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        raise


def prepare_volatility_model_data(df, target_col='Volatility_21d', forecast_horizon=21):
    feature_cols = [
        'Returns', 'Volatility_5d', 'Log_Price', 'Price_Change',
        'SMA_10', 'SMA_30', 'EMA_10', 'ATR_14', 'Volume_Change', 
        'Volume_MA10', 'Abs_Returns', 'Returns_MA5', 'Returns_Std5', 'RSI'
    ]
    
    df[f'Future_{target_col}'] = df[target_col].shift(-forecast_horizon)
    
    df.dropna(subset=[f'Future_{target_col}'], inplace=True)
    
    # Prepare features and target
    X = df[feature_cols].values
    y = df[f'Future_{target_col}'].values.reshape(-1, 1)
    
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_cols


def evaluate_volatility_model(model, X_test, y_test, scaler_y):
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)
    
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R-squared: {r2:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual Volatility')
    plt.plot(y_pred, label='Predicted Volatility')
    plt.title('Actual vs Predicted Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot prediction error
    plt.figure(figsize=(12, 6))
    plt.scatter(y_actual, y_pred)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--')
    plt.xlabel('Actual Volatility')
    plt.ylabel('Predicted Volatility')
    plt.title('Volatility Prediction Scatter Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return mse, r2


if __name__ == "__main__":
    df = load_and_preprocess_spy_data()
    
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = prepare_volatility_model_data(
        df, target_col='Volatility_21d', forecast_horizon=21
    )
    
    input_dim = X_train.shape[1]
    nn_model = NeuralNetworkFromScratch(
        layer_sizes=[input_dim, 32, 16, 1],
        activation='relu',
        learning_rate=0.001,
        l2_reg=0.0005
    )
    
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    loss_history = nn_model.train(
        X_train, y_train,
        epochs=2000,
        batch_size=32,
        validation_data=(X_val, y_val),
        patience=100,
        verbose=True
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    mse, r2 = evaluate_volatility_model(nn_model, X_test, y_test, scaler_y)
    
    print("\nFeature correlation with target volatility:")
    target_col = 'Future_Volatility_21d'
    correlations = df[feature_names + [target_col]].corr()[target_col].sort_values(ascending=False)
    print(correlations)
    
    latest_data = X_test[-1:].copy()  
    volatility_forecast = nn_model.predict(latest_data)
    volatility_forecast = scaler_y.inverse_transform(volatility_forecast)[0][0]
    
    print(f"\nVolatility forecast for the next 21 trading days: {volatility_forecast:.4f}")
    print(f"Annualized volatility: {volatility_forecast * 100:.2f}%")