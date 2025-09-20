import time
import random
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Keep your existing neural network functions

def generate_training_data(size=10000, a_range=(10, 100), b_range=(50, 500), m_range=(500, 1000)):
    """
    Generate training data for modular arithmetic predictions
    
    Parameters:
    - size: Number of samples to generate
    - a_range: Range for base values (min, max)
    - b_range: Range for exponent values (min, max)
    - m_range: Range for modulus values (min, max)
    """
    X = []
    y = []
    
    a_min, a_max = a_range
    b_min, b_max = b_range
    m_min, m_max = m_range
    
    for _ in range(size):
        a = np.random.randint(a_min, a_max)
        b = np.random.randint(b_min, b_max)
        m = np.random.randint(m_min, m_max)
        
        # Calculate a^b mod m
        result = pow(a, b, m)
        
        X.append([a, b, m])
        y.append(result)
    
    return np.array(X), np.array(y)

def create_and_train_model(X, y, validation_split=0.2, epochs=30, batch_size=64):
    """Create and train a neural network for precise modular arithmetic prediction."""
    # Normalize inputs
    X_norm = X.copy()
    X_norm[:, 0] = X_norm[:, 0] / 100.0  # Normalize a
    X_norm[:, 1] = X_norm[:, 1] / 500.0  # Normalize b
    X_norm[:, 2] = X_norm[:, 2] / 1000.0  # Normalize m
    
    # Normalize outputs relative to m
    y_norm = y / X[:, 2]
    
    # Create a deeper and wider model for better accuracy
    model = Sequential([
        Dense(256, activation='relu', input_shape=(3,)),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Output in [0, 1) range
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    
    # Train with more epochs for better convergence
    history = model.fit(
        X_norm, y_norm,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Prediction function with error correction
    def predict_modular_result(input_data):
        input_norm = input_data.copy()
        input_norm[:, 0] = input_norm[:, 0] / 100.0
        input_norm[:, 1] = input_norm[:, 1] / 500.0
        input_norm[:, 2] = input_norm[:, 2] / 1000.0
        
        pred_norm = model.predict(input_norm, verbose=0)
        predictions = pred_norm * input_data[:, 2]  # Denormalize
        
        # Round to nearest integer and ensure within modulus range
        predictions = np.round(predictions).astype(int)
        predictions = predictions % input_data[:, 2]
        
        return predictions
    
    return model, predict_modular_result, history

def plot_training_history(history):
    """Plot the training history of the model"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    return plt

# Add the missing Miner classes here

class BaseMiner:
    """Base class for all miners"""
    def __init__(self, miner_id):
        self.miner_id = miner_id
        self.blocks_mined = 0
        self.total_mining_time = 0
        # Default modular arithmetic parameters
        self.a_base = 7
        self.b_exp = 13
        self.m_mod = 23
    
    def set_modular_params(self, a, b, m):
        """Set the modular arithmetic parameters"""
        self.a_base = a
        self.b_exp = b
        self.m_mod = m
    
    def calculate_hash(self, previous_hash, proof):
        """Calculate a hash given a previous hash and proof of work"""
        guess = f'{previous_hash}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash
    
    def validate_proof(self, previous_hash, proof, difficulty):
        """Validate a proof of work and check if it meets the difficulty requirement"""
        guess_hash = self.calculate_hash(previous_hash, proof)
        return guess_hash[:difficulty] == '0' * difficulty
    
    def mine_block(self, previous_hash, difficulty):
        """Mine a new block - this method should be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement mine_block")


class TraditionalMiner(BaseMiner):
    """Traditional blockchain miner using brute force"""
    def __init__(self, miner_id):
        super().__init__(miner_id)
    
    def mine_block(self, previous_hash, difficulty):
        """Mine a new block using traditional brute force method"""
        proof = 0
        steps = []
        
        start_time = time.time()
        
        # Continue trying different proofs until a valid one is found
        while not self.validate_proof(previous_hash, proof, difficulty):
            # For visualization, record the current state every few steps
            if proof % 50 == 0 or proof < 10:
                # Calculate the actual result for modular arithmetic
                a, b, m = self.a_base, self.b_exp, self.m_mod
                
                # Time the traditional calculation
                calc_start_time = time.time()
                result = pow(a, b, m)
                calc_time = time.time() - calc_start_time
                
                # Record the current mining time
                current_mining_time = time.time() - start_time
                
                # Record this step for visualization
                steps.append({
                    'proof': proof,
                    'a': a,
                    'b': b,
                    'm': m,
                    'result': result,
                    'traditional_time': calc_time,
                    'mining_time': current_mining_time
                })
            
            proof += 1
        
        # Record final step
        end_time = time.time()
        mining_time = end_time - start_time
        self.blocks_mined += 1
        self.total_mining_time += mining_time
        
        # Record final mining details
        steps.append({
            'proof': proof,
            'mining_time': mining_time,
            'status': 'Success'
        })
        
        return proof, steps


class NeuralMiner(BaseMiner):
    """Blockchain miner enhanced with neural network predictions"""
    def __init__(self, miner_id, predict_fn):
        super().__init__(miner_id)
        self.predict_fn = predict_fn
    
    def mine_block(self, previous_hash, difficulty):
        """Mine a new block using neural network predictions to optimize proof search."""
        steps = []
        start_time = time.time()
        
        # Use neural network to predict a starting proof based on modular arithmetic
        a, b, m = self.a_base, self.b_exp, self.m_mod
        input_data = np.array([[a, b, m]])
        nn_start_time = time.time()
        predicted_proof = int(self.predict_fn(input_data)[0])  # Get neural prediction
        nn_time = time.time() - nn_start_time
        
        # Start proof search from the predicted value
        proof = max(0, predicted_proof - 10)  # Start slightly below prediction to account for error
        
        # Calculate the actual result for comparison
        trad_start_time = time.time()
        actual_result = pow(a, b, m)
        traditional_time = time.time() - trad_start_time
        
        # Record initial prediction step
        steps.append({
            'proof': proof,
            'a': a,
            'b': b,
            'm': m,
            'nn_prediction': float(predicted_proof),
            'actual_result': actual_result,
            'nn_time': nn_time,
            'traditional_time': traditional_time,
            'mining_time': time.time() - start_time
        })
        
        # Search for valid proof starting from neural prediction
        while not self.validate_proof(previous_hash, proof, difficulty):
            if proof % 50 == 0 or proof < predicted_proof + 10:
                steps.append({
                    'proof': proof,
                    'a': a,
                    'b': b,
                    'm': m,
                    'nn_prediction': float(predicted_proof),
                    'actual_result': actual_result,
                    'nn_time': nn_time,
                    'traditional_time': traditional_time,
                    'mining_time': time.time() - start_time
                })
            proof += 1
        
        # Record final step
        end_time = time.time()
        mining_time = end_time - start_time
        self.blocks_mined += 1  # Fixed: was self.keys_mined
        self.total_mining_time += mining_time
        
        steps.append({
            'proof': proof,
            'mining_time': mining_time,
            'status': 'Success'
        })
        
        return proof, steps

if __name__ == "__main__":
    # Generate data
    X, y = generate_training_data(20000)
    
    # Train model
    model, predict_fn, history = create_and_train_model(X, y, epochs=30)
    
    # Save model
    model.save('modular_math_model.h5')
    
    # Plot and save training history
    plot = plot_training_history(history)
    plot.savefig('training_history.png')
    
    # Test a few predictions
    test_cases = np.array([
        [15, 200, 700],
        [50, 300, 800],
        [25, 150, 900]
    ])
    
    predictions = predict_fn(test_cases)
    
    for i, pred in enumerate(predictions):
        a, b, m = test_cases[i]
        actual = pow(a, b, m)
        print(f"Case {i+1}: {a}^{b} mod {m}")
        print(f"  Prediction: {int(pred[0])}")
        print(f"  Actual: {actual}")
        print(f"  Error: {abs(int(pred[0]) - actual)}")
        print()