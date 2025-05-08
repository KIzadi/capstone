import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Will not be able to use as a fallback for data loading.")

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
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for weights
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
            
        # History for tracking loss during training
        self.loss_history = []
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
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
            
            # Use linear activation for the output layer
            if i == len(self.weights) - 1:
                A = Z  # Linear activation for regression
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
        
        # Backpropagate the error
        for l in range(len(self.weights) - 1, -1, -1):
            gradients_w[l] = (1/m) * np.dot(activations[l].T, delta) + (self.l2_reg * self.weights[l])
            gradients_b[l] = (1/m) * np.sum(delta, axis=0, keepdims=True)
            
            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.activation_derivative(Z_values[l-1])
                
        return gradients_w, gradients_b
    
    def train(self, X, y, epochs=1000, batch_size=32, validation_data=None, patience=50, verbose=True):
        """
        Train the neural network using mini-batch gradient descent.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input training data of shape (n_samples, n_features)
        y : numpy.ndarray
            Target training values of shape (n_samples, n_outputs)
        epochs : int
            Number of training epochs
        batch_size : int
            Size of mini-batches
        validation_data : tuple
            Tuple of (X_val, y_val) for validation
        patience : int
            Number of epochs with no improvement after which training will be stopped
        verbose : bool
            Whether to print progress during training
            
        Returns:
        --------
        loss_history : list
            List of training losses for each epoch
        """
        n_samples = X.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                activations, Z_values = self.forward(X_batch)
                
                # Backward pass
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations, Z_values)
                
                # Update weights and biases
                for l in range(len(self.weights)):
                    self.weights[l] -= self.learning_rate * gradients_w[l]
                    self.biases[l] -= self.learning_rate * gradients_b[l]
            
            # Calculate loss for the full training set
            activations, _ = self.forward(X)
            y_pred = activations[-1]
            train_loss = np.mean((y_pred - y)**2)
            self.loss_history.append(train_loss)
            
            # Validation if provided
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
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                val_msg = f", Validation Loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.6f}{val_msg}")
                
        return self.loss_history
    
    def predict(self, X):
        """
        Make predictions using the trained neural network.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Predictions of shape (n_samples, n_outputs)
        """
        activations, _ = self.forward(X)
        return activations[-1]


def load_and_preprocess_spy_data(file_path="spy_stock_data.csv"):
    """
    Load and preprocess SPY stock data for volatility prediction.
    
    Parameters:
    -----------
    file_path : str
        Path to the SPY stock data CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Preprocessed DataFrame with features and target variable
    """
    try:
        # Define potential paths to check
        potential_paths = [
            file_path,
            f"capstone_data/{file_path}",
            f"../capstone_data/{file_path}",
            f"../../capstone_data/{file_path}",
            r'C:\Users\Kevin\Documents\cappystone\capstone\capstone_data\spy_stock_data.csv'
        ]
        
        # Try each path until one works
        for path in potential_paths:
            try:
                print(f"Attempting to load data from: {path}")
                df = pd.read_csv(path, skiprows=2)
                print(f"Successfully loaded data from: {path}")
                break
            except FileNotFoundError:
                continue
        else:
            # If no file found, use yfinance to download data
            print("No local data file found. Downloading SPY data from Yahoo Finance...")
            try:
                import yfinance as yf
                df = yf.download("SPY", start="2010-01-01", end="2023-12-31", progress=False)
                df.reset_index(inplace=True)
                print("Successfully downloaded SPY data from Yahoo Finance.")
                # Set column names to match expected format
                df.columns = ['Date'] + list(df.columns[1:])
            except ImportError:
                raise ImportError("yfinance is required to download data. Please install it or provide a valid data file.")
            except Exception as e:
                raise Exception(f"Failed to download data: {e}")
        
        # Map the actual columns to the expected column names if needed
        if 'Date' in df.columns and 'Close' in df.columns:
            # Data already has correct column names (likely from yfinance)
            pass
        else:
            # Try to map columns from a CSV file with unknown structure
            column_mapping = {
                df.columns[0]: "Date",
                df.columns[1]: "Close",
                df.columns[2]: "High", 
                df.columns[3]: "Low",
                df.columns[4]: "Open",
                df.columns[5]: "Volume"
            }
            df = df.rename(columns=column_mapping)
        
        # Convert Date to datetime and set as index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        
        # Ensure all price and volume data is numeric
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Data loaded successfully: {df.shape[0]} rows from {df.index.min()} to {df.index.max()}")
        
        # Calculate returns and historical volatility
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate realized volatility for different time windows (annualized)
        df['Volatility_5d'] = df['Returns'].rolling(window=5).std() * np.sqrt(252)
        df['Volatility_21d'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)  # ~1 month
        df['Volatility_63d'] = df['Returns'].rolling(window=63).std() * np.sqrt(252)  # ~3 months
        
        # Feature engineering for volatility prediction
        # 1. Price-based features
        df['Log_Price'] = np.log(df['Close'])
        df['Price_Change'] = df['Close'].pct_change(5)
        
        # 2. Moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        
        # 3. Volatility indicators
        # True Range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR_14'] = df['TR'].rolling(window=14).mean()
        
        # 4. Volume-based features
        df['Volume_Change'] = df['Volume'].pct_change(5)
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
        
        # 5. Return-based features
        df['Abs_Returns'] = np.abs(df['Returns'])
        df['Returns_MA5'] = df['Returns'].rolling(window=5).mean()
        df['Returns_Std5'] = df['Returns'].rolling(window=5).std()
        
        # 6. Momentum indicators
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Drop NA values created by rolling windows
        df.dropna(inplace=True)
        
        print(f"Preprocessed data shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        raise


def prepare_volatility_model_data(df, target_col='Volatility_21d', forecast_horizon=21):
    """
    Prepare data for volatility prediction model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed DataFrame with features and target
    target_col : str
        Column name for the target volatility
    forecast_horizon : int
        Number of days ahead to forecast volatility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaler_X, scaler_y : tuple
        Prepared and scaled data for training and testing
    feature_names : list
        Names of features used in the model
    """
    # Select features for prediction
    feature_cols = [
        'Returns', 'Volatility_5d', 'Log_Price', 'Price_Change',
        'SMA_10', 'SMA_30', 'EMA_10', 'ATR_14', 'Volume_Change', 
        'Volume_MA10', 'Abs_Returns', 'Returns_MA5', 'Returns_Std5', 'RSI'
    ]
    
    # Create future target (shifted volatility)
    df[f'Future_{target_col}'] = df[target_col].shift(-forecast_horizon)
    
    # Drop rows with NA in the target
    df.dropna(subset=[f'Future_{target_col}'], inplace=True)
    
    # Prepare features and target
    X = df[feature_cols].values
    y = df[f'Future_{target_col}'].values.reshape(-1, 1)
    
    # Split into training and testing sets, preserving time order
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Standardize features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    # Standardize target
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_cols


def evaluate_volatility_model(model, X_test, y_test, scaler_y, df, feature_names):
    """
    Evaluate the volatility prediction model.
    
    Parameters:
    -----------
    model : NeuralNetworkFromScratch
        Trained neural network model
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Scaled test targets
    scaler_y : StandardScaler
        Scaler used for the target variable
    df : pandas.DataFrame
        DataFrame containing the original data
    feature_names : list
        Names of features used in the model
        
    Returns:
    --------
    mse : float
        Mean squared error
    r2 : float
        R-squared value
    """
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)
    
    # Extract dates for the test set
    test_dates = df.index[-len(y_test):]
    
    # Calculate metrics
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R-squared: {r2:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    
    # Plot actual vs predicted with dates
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_actual, label='Actual Volatility', color='#1f77b4', linewidth=2)
    plt.plot(test_dates, y_pred, label='Predicted Volatility', color='#d62728', linestyle='-', linewidth=1.5)
    plt.title('Actual vs Predicted Volatility', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility', fontsize=12)
    
    # Format the date ticks
    plt.gcf().autofmt_xdate()  # Auto-format the x-axis date labels
    
    # Add grid and legend with better visibility
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Set y-axis limits with a small buffer
    max_val = max(np.max(y_actual), np.max(y_pred))
    min_val = min(np.min(y_actual), np.min(y_pred))
    buffer = (max_val - min_val) * 0.1
    plt.ylim(min_val - buffer, max_val + buffer)
    
    # Format date ticks more clearly
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
    plt.xticks(rotation=45)
    
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
    # 1. Load and preprocess data
    df = load_and_preprocess_spy_data()
    
    # 2. Prepare data for the volatility prediction model
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names = prepare_volatility_model_data(
        df, target_col='Volatility_21d', forecast_horizon=21
    )
    
    # 3. Create and train the neural network
    # Input layer: number of features
    # Hidden layers: [32, 16] - two hidden layers with 32 and 16 neurons
    # Output layer: 1 (volatility prediction)
    input_dim = X_train.shape[1]
    nn_model = NeuralNetworkFromScratch(
        layer_sizes=[input_dim, 32, 16, 1],
        activation='relu',
        learning_rate=0.001,
        l2_reg=0.0005
    )
    
    # 4. Train the model with early stopping using validation data
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
    
    # 5. Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 6. Evaluate the model
    mse, r2 = evaluate_volatility_model(nn_model, X_test, y_test, scaler_y, df, feature_names)
    
    # 7. Feature importance analysis
    # We can't directly get feature importance from a neural network
    # But we can analyze correlation of features with target
    print("\nFeature correlation with target volatility:")
    target_col = 'Future_Volatility_21d'
    correlations = df[feature_names + [target_col]].corr()[target_col].sort_values(ascending=False)
    print(correlations)
    
    # 8. Make a volatility forecast for the next period
    latest_data = X_test[-1:].copy()  # Get the most recent data point
    volatility_forecast = nn_model.predict(latest_data)
    volatility_forecast = scaler_y.inverse_transform(volatility_forecast)[0][0]
    
    print(f"\nVolatility forecast for the next 21 trading days: {volatility_forecast:.4f}")
    print(f"Annualized volatility: {volatility_forecast * 100:.2f}%")