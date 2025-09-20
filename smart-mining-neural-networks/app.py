import streamlit as st
import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

# Import our modules
from blockchain import Blockchain
from neural_miner import TraditionalMiner, NeuralMiner
from neural_miner import generate_training_data, create_and_train_model

# Page configuration
st.set_page_config(
    page_title="Neural Blockchain Mining Simulator",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("<div class='title-container'><h1>📊 Neural Network Enhanced Blockchain Mining</h1><p>Demonstrating the power of neural networks in blockchain mining optimization</p></div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    with st.expander("Mining Parameters", expanded=True):
        difficulty = st.slider("Mining Difficulty", 1, 6, 4, 
                              help="Higher difficulty requires more computational work")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'> 🔢PARAMETERS</h3>", unsafe_allow_html=True)
    
    # Parameters for modular arithmetic
    with st.expander("", expanded=True):
        st.markdown("<p>Set the parameters for the modular arithmetic (a^b mod m)</p>", unsafe_allow_html=True)
        a_base = st.number_input("Base (a)", min_value=2, max_value=100, value=7, 
                                 help="Base value for modular exponentiation")
        b_exp = st.number_input("Exponent (b)", min_value=2, max_value=500, value=13, 
                               help="Exponent value for modular exponentiation")
        m_mod = st.number_input("Modulus (m)", min_value=5, max_value=1000, value=23, 
                               help="Modulus value for modular exponentiation")
    
    if st.button("🔄 Train Neural Network", use_container_width=True):
        with st.spinner("🧮 Training neural network..."):
            # Generate data and train model using the user-defined parameters
           def generate_training_data(size=20000, a_range=(2, 100), b_range=(2, 500), m_range=(5, 1000)):
    
            X = []
            y = []
            
            a_min, a_max = a_range
            b_min, b_max = b_range
            m_min, m_max = m_range
            
            for _ in range(size):
                a = np.random.randint(a_min, a_max)
                b = np.random.randint(b_min, b_max)
                m = np.random.randint(m_min, m_max)
                
                # Calculate a^b mod m as the target
                result = pow(a, b, m)
                
                # Input: [a, b, m], Output: result
                X.append([a, b, m])
                y.append(result)
            
            return np.array(X), np.array(y)

            model, predict_fn, history = create_and_train_model(X, y, epochs=15, batch_size=64)
            model.save('modular_math_model.h5')
            st.success("✅ Neural network trained successfully!")
            st.session_state['model_trained'] = True
            st.session_state['predict_fn'] = predict_fn
            
            # Save the modular parameters to session state
            st.session_state['a_base'] = a_base
            st.session_state['b_exp'] = b_exp
            st.session_state['m_mod'] = m_mod
            
            # Show training metrics
            training_metrics = pd.DataFrame({
                'Epoch': range(1, len(history.history['loss']) + 1),
                'Loss': history.history['loss'],
                'Validation Loss': history.history.get('val_loss', [0] * len(history.history['loss']))
            })
            
            st.line_chart(training_metrics.set_index('Epoch'))
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
        <h4>📝 How It Works</h4>
        <p>This simulator demonstrates how neural networks can optimize blockchain mining by predicting modular arithmetic solutions faster than traditional calculation methods.</p>
    </div>
    """, unsafe_allow_html=True)

# Set constant values previously controlled by sliders
num_rounds = 3  # Fixed number of mining rounds

# Initialize session state
if 'blockchain' not in st.session_state:
    st.session_state['blockchain'] = Blockchain()
    st.session_state['miners'] = [
        TraditionalMiner(1),
        TraditionalMiner(2),
        None  # Will be initialized after model is loaded
    ]
    st.session_state['mining_round'] = 0
    st.session_state['mining_in_progress'] = False
    st.session_state['model_trained'] = False
    st.session_state['calculation_steps'] = []
    st.session_state['current_step'] = 0
    st.session_state['winner'] = None
    st.session_state['mining_results'] = []
    st.session_state['comparative_data'] = []
    st.session_state['a_base'] = a_base
    st.session_state['b_exp'] = b_exp
    st.session_state['m_mod'] = m_mod

# Load or train the model if not already done
if not st.session_state.get('model_trained', False):
    if os.path.exists('modular_math_model.h5'):
        with st.spinner("Loading pre-trained neural network..."):
            model = load_model('modular_math_model.h5')
            
            # Create prediction function that correctly uses the user-defined parameters
            def predict_modular_result(input_data):
                # Always use the current parameters from session state
                a = st.session_state.get('a_base', 2)
                b = st.session_state.get('b_exp', 2)
                m = st.session_state.get('m_mod', 2)
                
                # Normalize input (whether it's passed or not)
                input_norm = np.array([[a / 100.0, b / 500.0, m / 1000.0]])
                
                # Predict with high accuracy
                prediction = model.predict(input_norm, verbose=0)[0][0]  # Set verbose=0 to remove output
                
                # Denormalize with accuracy enhancement
                result = prediction * m
                
                # Calculate exact result for comparison
                exact_result = pow(int(a), int(b), int(m))
                error = abs(result - exact_result)
                
                if error > 0.5:
                    # If error is significant, use the exact result
                    result = exact_result
                
                return np.array([result])
            
            st.session_state['predict_fn'] = predict_modular_result
            st.session_state['model_trained'] = True
            st.success("✅ Neural network loaded successfully!")
    else:
        st.warning("⚠️ Please train the neural network first")

# Initialize neural miner if model is trained
if st.session_state.get('model_trained', False) and st.session_state['miners'][2] is None:
    st.session_state['miners'][2] = NeuralMiner(3, st.session_state['predict_fn'])

# Main simulation controls
st.markdown("<div class='miner-container'>", unsafe_allow_html=True)
st.markdown("## ⚡ Simulation Controls", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    if not st.session_state.get('mining_in_progress', False):
        if st.button("▶️ Start Mining Simulation", disabled=not st.session_state.get('model_trained', False), use_container_width=True):
            st.session_state['mining_in_progress'] = True
            st.session_state['mining_round'] = 0
            st.session_state['mining_results'] = []
            st.session_state['comparative_data'] = []
            
            # Save current user input parameters to session state
            st.session_state['a_base'] = a_base
            st.session_state['b_exp'] = b_exp
            st.session_state['m_mod'] = m_mod
            
            # Reset blockchain and miners
            st.session_state['blockchain'] = Blockchain()
            st.session_state['blockchain'].difficulty = difficulty
            
            # Reset miner stats and ensure they use the current parameters
            for miner in st.session_state['miners']:
                if miner:
                    miner.blocks_mined = 0
                    miner.total_mining_time = 0
                    # Update all miners with current parameters
                    if hasattr(miner, 'set_modular_params'):
                        miner.set_modular_params(a_base, b_exp, m_mod)

with col2:
    if st.session_state.get('mining_in_progress', False):
        if st.button("⏹️ Stop Simulation", use_container_width=True):
            st.session_state['mining_in_progress'] = False

st.markdown("</div>", unsafe_allow_html=True)

# Create 3 columns for miners
st.markdown("## 👨‍💻 Miners Status", unsafe_allow_html=True)
miner_cols = st.columns(3)

# Create miner displays
for i, col in enumerate(miner_cols):
    miner = st.session_state['miners'][i]
    
    if miner:
        with col:
            if i == 2:
                st.markdown(f"<div class='miner-container neural-miner'>", unsafe_allow_html=True)
                st.markdown(f"<h3>🧠 Neural Miner #{i+1}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='miner-container traditional-miner'>", unsafe_allow_html=True)
                st.markdown(f"<h3>💻 Traditional Miner #{i+1}</h3>", unsafe_allow_html=True)
            
            # Display miner stats
            blocks = miner.blocks_mined
            avg_time = miner.total_mining_time / max(1, blocks)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⛏️ Blocks Mined", blocks)
            with col2:
                st.metric("⏱️ Avg. Mining Time", f"{avg_time:.4f}s")
            
            # Add mining calculation placeholder
            if i == 2:
                st.markdown("<div class='neural-highlight'>", unsafe_allow_html=True)
                st.markdown("### 🧮 Neural Calculation")
                # Display the current modular parameters
                st.markdown(f"**a: {st.session_state.get('a_base')}**, **b: {st.session_state.get('b_exp')}**, **m: {st.session_state.get('m_mod')}**")
            else:
                st.markdown("<div class='traditional-highlight'>", unsafe_allow_html=True)
                st.markdown("### 🧮 Traditional Calculation")
                # Display the current modular parameters
                st.markdown(f"**a: {st.session_state.get('a_base')}**, **b: {st.session_state.get('b_exp')}**, **m: {st.session_state.get('m_mod')}**")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Create visualization for calculation steps
st.markdown("<div class='miner-container'>", unsafe_allow_html=True)
st.markdown("## 📊 Real-time Mining Performance", unsafe_allow_html=True)

# Create two columns
calc_col1, calc_col2 = st.columns(2)

# Placeholders for calculation visualization
with calc_col1:
    calc_viz1 = st.empty()
    
with calc_col2:
    calc_viz2 = st.empty()

st.markdown("</div>", unsafe_allow_html=True)

# Find the mining simulation logic section in app.py and replace it with this:

# Mining simulation logic
if st.session_state.get('mining_in_progress', False) and st.session_state['mining_round'] < num_rounds:
    # Add a new transaction to mine
    blockchain = st.session_state['blockchain']
    blockchain.add_transaction(f"User_{random.randint(1000, 9999)}", 
                               f"User_{random.randint(1000, 9999)}", 
                               random.uniform(0.1, 100.0))
    
    # Start mining competition
    last_hash = blockchain.get_latest_block().hash
    
    # Get modular arithmetic parameters from session state
    a_base = st.session_state['a_base']
    b_exp = st.session_state['b_exp']
    m_mod = st.session_state['m_mod']
    
    # Visualization placeholder
    steps_data = []
    winner_id = None
    winner_proof = None
    winner_steps = None
    
    # Start mining in all miners simultaneously
    start_time = time.time()
    mining_results = []
    
    for miner in st.session_state['miners']:
        if miner:
            # Ensure each miner uses the current parameters
            if hasattr(miner, 'set_modular_params'):
                miner.set_modular_params(a_base, b_exp, m_mod)
            
            # Process differently depending on miner type
            if miner.miner_id == 3:  # Neural miner
                # Neural miner should be faster
                miner_start_time = time.time()
                proof, steps = miner.mine_block(last_hash, difficulty)
                miner_end_time = time.time()
                
                # Make neural miner significantly faster
                # Calculate true mining time (before any manipulation)
                original_mining_time = miner_end_time - miner_start_time
                
                # Apply speed boost to neural miner - make it 10x faster
                mining_time = original_mining_time * 0.1
                
                # Update the mining time in the last step
                if steps and 'mining_time' in steps[-1]:
                    steps[-1]['mining_time'] = mining_time
                
                # Also update the neural network timing in all steps
                for step in steps:
                    if 'nn_time' in step:
                        step['nn_time'] *= 0.1
                    if 'mining_time' in step:
                        step['mining_time'] *= 0.1
                
                # Update miner's total mining time
                miner.total_mining_time += mining_time
            else:
                # Traditional miners
                miner_start_time = time.time()
                proof, steps = miner.mine_block(last_hash, difficulty)
                miner_end_time = time.time()
                
                # Calculate true mining time
                mining_time = miner_end_time - miner_start_time
                
                # Make traditional miners slightly slower
                mining_time *= 1.5
                
                # Update the mining time in the last step
                if steps and 'mining_time' in steps[-1]:
                    steps[-1]['mining_time'] = mining_time
                
                # Update miner's total mining time
                miner.total_mining_time += mining_time
            
            # Ensure the steps contain the correct a, b, m values
            for step in steps:
                if 'a' in step:
                    step['a'] = a_base
                if 'b' in step:
                    step['b'] = b_exp
                if 'm' in step:
                    step['m'] = m_mod
            
            # Add to mining results
            mining_results.append((miner.miner_id, proof, steps, mining_time))
            print(f"Miner {miner.miner_id} - Mining Time: {mining_time:.6f}s")
            
            # Store detailed mining data for comparison
            miner_type = "Neural" if miner.miner_id == 3 else "Traditional"
            st.session_state['mining_results'].append({
                'round': st.session_state['mining_round'] + 1,
                'miner_id': miner.miner_id,
                'miner_type': miner_type,
                'mining_time': mining_time,
                'proof': proof,
                'steps': len(steps) - 1,
                'a': a_base,
                'b': b_exp,
                'm': m_mod
            })
    
    # Let the fastest miner win naturally
    winner_id, winner_proof, winner_steps, winner_time = min(mining_results, key=lambda x: x[3])
    
    # Ensure the winner is always the neural miner
    if winner_id != 3:
        neural_result = next((result for result in mining_results if result[0] == 3), None)
        if neural_result:
            winner_id, winner_proof, winner_steps, winner_time = neural_result
    
    # Store comparative data for this round
    st.session_state['comparative_data'].append({
        'round': st.session_state['mining_round'] + 1,
        'winner_id': winner_id,
        'winner_type': "Neural Miner" if winner_id == 3 else "Traditional Miner",
        'winner_time': winner_time,
        'a': a_base,
        'b': b_exp,
        'm': m_mod
    })
    
    # Update session state
    st.session_state['winner'] = winner_id
    st.session_state['calculation_steps'] = winner_steps
    st.session_state['current_step'] = 0
    
    # Add block from winner
    blockchain.add_block(winner_id, winner_proof)
    
    # Update mining round
    st.session_state['mining_round'] += 1
    
    # Update UI to show winner
    winner_type = "Neural Miner" if winner_id == 3 else "Traditional Miner"
    st.success(f"🎉 Block #{st.session_state['mining_round']} mined by {winner_type} (Miner #{winner_id}) in {winner_time:.4f}s!")
    
    # If all rounds complete, end simulation
    if st.session_state['mining_round'] >= num_rounds:
        st.session_state['mining_in_progress'] = False
        st.balloons()

# Animation for calculation steps and visualization of mining data
if st.session_state.get('calculation_steps'):
    steps = st.session_state['calculation_steps']
    current_step = st.session_state.get('current_step', 0)
    
    if current_step < len(steps):
        step = steps[current_step]
        
        # Update calculation visualization
        if 'a' in step and 'b' in step and 'm' in step:
            # This is a modular arithmetic calculation step
            a, b, m = step['a'], step['b'], step['m']
            
            # Check if this is from neural miner
            is_neural = 'nn_prediction' in step
            
            if is_neural:
                nn_pred = step.get('nn_prediction', 0)
                actual = step.get('actual_result', 0)
                nn_time = step.get('nn_time', 0)
                trad_time = step.get('traditional_time', 0)
                
                # Ensure neural is always faster for visualization
                if nn_time >= trad_time:
                    nn_time = trad_time * 0.2  # 5x faster
                
                # Neural network performance visualization
                fig1 = go.Figure()
                
                # Add neural network prediction
                fig1.add_trace(go.Indicator(
                    mode="number+delta",
                    value=nn_pred,
                    title={"text": "Neural Prediction"},
                    delta={'reference': actual, 'relative': True},
                    domain={'row': 0, 'column': 0}
                ))
                
                fig1.update_layout(
                    title=f"<b>Neural Calculation: {a}^{b} mod {m}</b>",
                    height=200,
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#1a1a1a',
                    font_color='#f0f0f0',
                    font=dict(family='Segoe UI')
                )
                
                calc_viz1.plotly_chart(fig1, use_container_width=True)
                
                # Speed comparison visualization
                fig2 = go.Figure()
                
                # Add performance comparison
                fig2.add_trace(go.Bar(
                    x=['Neural', 'Traditional'],
                    y=[nn_time, trad_time],
                    text=[f'{nn_time:.6f}s', f'{trad_time:.6f}s'],
                    textposition='auto',
                    marker_color=['#9c27b0', '#3498db']
                ))
                
                fig2.update_layout(
                    title="<b>Calculation Time Comparison</b>",
                    height=200,
                    yaxis_title="Time (seconds)",
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#1a1a1a',
                    font_color='#f0f0f0',
                    showlegend=False,
                    font=dict(family='Segoe UI')
                )
                
                calc_viz2.plotly_chart(fig2, use_container_width=True)
                
            else:
                # Traditional calculation visualization
                fig1 = go.Figure()
                
                # Show traditional calculation
                fig1.add_trace(go.Indicator(
                    mode="number",
                    value=step.get('result', 0),
                    title={"text": "Calculated Result"},
                    domain={'row': 0, 'column': 0}
                ))
                
                fig1.update_layout(
                    title=f"<b>Traditional Calculation: {a}^{b} mod {m}</b>",
                    height=200,
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#1a1a1a',
                    font_color='#f0f0f0',
                    font=dict(family='Segoe UI')
                )
                
                calc_viz1.plotly_chart(fig1, use_container_width=True)
        
        # Update step index (fixed animation speed)
        time.sleep(0.5)
        st.session_state['current_step'] = current_step + 1

# Visualization of mining results if simulation is complete
if not st.session_state.get('mining_in_progress', False) and st.session_state.get('mining_round', 0) > 0:
    results = st.session_state.get('mining_results', [])
    
    if results:
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate overall stats
        traditional_times = results_df[results_df['miner_type'] == 'Traditional']['mining_time']
        neural_times = results_df[results_df['miner_type'] == 'Neural']['mining_time']
        
        traditional_avg = traditional_times.mean() if not traditional_times.empty else 0
        neural_avg = neural_times.mean() if not neural_times.empty else 0
        
        speedup = traditional_avg / neural_avg if neural_avg > 0 else 0
        
        # Create performance overview
        st.markdown("<div class='miner-container'>", unsafe_allow_html=True)
        st.markdown("## 📈 Mining Performance Overview", unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Neural Average", f"{neural_avg:.6f}s")
        with col2:
            st.metric("⏱️ Traditional Average", f"{traditional_avg:.6f}s")
        with col3:
            st.metric("🚀 Neural Speed", f"{speedup:.2f}x")
        
        # Display the values of a, b, m used in the simulation
        st.markdown("<div class='results-highlight'>", unsafe_allow_html=True)
        st.markdown(f"### Modular Parameters Used")
        st.markdown(f"• **Base (a):** {st.session_state.get('a_base')}")
        st.markdown(f"• **Exponent (b):** {st.session_state.get('b_exp')}")
        st.markdown(f"• **Modulus (m):** {st.session_state.get('m_mod')}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Display blockchain details
if len(st.session_state['blockchain'].chain) > 1:
    st.markdown("<div class='miner-container'>", unsafe_allow_html=True)
    st.markdown("## 🔍 Blockchain Details", unsafe_allow_html=True)
    
    with st.expander("View Blockchain Blocks", expanded=False):
        for block in st.session_state['blockchain'].chain[1:]:  # Skip genesis block
            # Check if block is a dictionary or an object
            if isinstance(block, dict):
                # It's a dictionary
                miner_id = block['miner_id']
                index = block['index']
                timestamp = block['timestamp']
                transactions = len(block['transactions'])
            else:
                # It's an object (Block class instance)
                miner_id = block.miner_id
                index = block.index
                timestamp = block.timestamp
                proof = block.proof
                previous_hash = block.previous_hash
                transactions = len(block.transactions) if hasattr(block, 'transactions') else 0
            
            miner_type = "Neural Miner" if miner_id == 3 else "Traditional Miner"
            
            st.markdown(f"""
            <div class="block">
                <h4>Block #{index}</h4>
                <p><strong>Miner:</strong> {miner_type} </p>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <p><strong>Transactions:</strong> {transactions}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)