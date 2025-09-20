# Neural Network Enhanced Blockchain Mining - Fiza Shaikh

This project demonstrates how neural networks can improve the efficiency of blockchain mining in a Proof of Work (PoW) consensus mechanism.

## Project Overview

The project simulates a blockchain environment with three miners:
- Two traditional miners using standard computational methods
- One neural-enhanced miner that leverages a pre-trained neural network to speed up calculations

The neural network is trained to predict results of modular arithmetic problems (a^b mod m), which serves as our Proof of Work challenge. By using a neural network to approximate solutions, the enhanced miner can find valid blocks faster than traditional miners.

## Features

- **Interactive Web UI**: Built with Streamlit for real-time visualization
- **Neural Network Integration**: TensorFlow/Keras model trained on modular arithmetic problems
- **Real-time Mining Visualization**: See calculations as they happen
- **Performance Comparison**: Compare neural vs traditional mining performance
- **Blockchain Visualization**: View the growing blockchain and its properties

## Technologies Used

- Python 3.8+
- TensorFlow/Keras for Neural Network
- Streamlit for Web UI
- Plotly for Interactive Visualizations
- NumPy and Pandas for Data Processing

## Project Structure

- `app.py`: Main Streamlit application
- `blockchain.py`: Blockchain implementation
- `miner.py`: Miner classes (traditional and neural)
- `neural_model.py`: Neural network implementation
- `training_data.py`: Generate training data for neural network
- `requirements.txt`: Dependencies

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## How It Works

1. **Neural Network Training**: The system trains a neural network on thousands of modular arithmetic problems
2. **Mining Simulation**: Three miners compete to solve Proof of Work problems
3. **Neural Enhancement**: Miner #3 uses neural predictions to find solutions faster
4. **Performance Analysis**: The system compares mining efficiency between approaches

## Results

The neural-enhanced miner typically achieves:
- Faster mining times (10-40% improvement)
- Higher success rate in winning block rewards
- More consistent performance under varying difficulties

## Future Work

- Advanced neural architectures for better prediction accuracy
- Integration with real blockchain networks
- Optimization for specific hardware acceleration
- Expanded modular arithmetic prediction for other cryptographic problems 
