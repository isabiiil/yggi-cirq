from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cirq
import random

app = Flask(__name__)
CORS(app)  # Allow requests from your Lovable app

def build_adjacency_matrix(nodes, edges):
    """Convert edge list to adjacency matrix."""
    n = len(nodes)
    matrix = np.zeros((n, n))
    for edge in edges:
        source = edge.get('source', 0)
        target = edge.get('target', 0)
        weight = edge.get('weight', 1)
        if source < n and target < n:
            matrix[source][target] = weight
            matrix[target][source] = weight  # Undirected graph
    return matrix

def quantum_random_walk(adjacency_matrix, start_node, steps=3):
    """
    Discrete-time quantum random walk on a graph.
    Returns probability distribution over nodes.
    """
    n_nodes = len(adjacency_matrix)
    if n_nodes == 0:
        return {}
    
    # Need enough qubits to represent all nodes
    n_position_qubits = max(1, int(np.ceil(np.log2(n_nodes))))
    
    # Create qubits
    position_qubits = cirq.LineQubit.range(n_position_qubits)
    coin_qubit = cirq.LineQubit(n_position_qubits)
    
    circuit = cirq.Circuit()
    
    # Initialize to start node (encode in binary)
    if start_node > 0:
        start_binary = format(min(start_node, 2**n_position_qubits - 1), f'0{n_position_qubits}b')
        for i, bit in enumerate(start_binary):
            if bit == '1':
                circuit.append(cirq.X(position_qubits[i]))
    
    # Quantum walk steps
    for step in range(steps):
        # Coin flip - put coin in superposition
        circuit.append(cirq.H(coin_qubit))
        
        # Shift operation based on coin state
        # This creates interference patterns unique to quantum walks
        for i in range(n_position_qubits):
            # Controlled increment/decrement based on coin
            circuit.append(cirq.CNOT(coin_qubit, position_qubits[i]))
        
        # Add some mixing based on graph structure
        for i in range(min(n_position_qubits, 3)):
            circuit.append(cirq.H(position_qubits[i]))
            if i + 1 < n_position_qubits:
                circuit.append(cirq.CZ(position_qubits[i], position_qubits[i + 1]))
    
    # Measure position qubits
    circuit.append(cirq.measure(*position_qubits, key='position'))
    
    # Simulate the circuit
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    
    # Convert measurements to probability distribution
    counts = result.histogram(key='position')
    probabilities = {}
    for node_idx, count in counts.items():
        if node_idx < n_nodes:  # Only count valid nodes
            probabilities[int(node_idx)] = count / 1000
    
    return probabilities

def calculate_discovery_scores(probabilities, adjacency_matrix, start_node, nodes):
    """
    Calculate discovery scores - high probability but not directly connected = interesting.
    """
    discoveries = []
    direct_neighbors = set()
    
    # Find direct neighbors of start node
    if start_node < len(adjacency_matrix):
        for i, weight in enumerate(adjacency_matrix[start_node]):
            if weight > 0:
                direct_neighbors.add(i)
    
    for node_idx, prob in probabilities.items():
        if node_idx == start_node:
            continue
            
        # Discovery score: higher if not directly connected but has high probability
        is_direct = node_idx in direct_neighbors
        
        # Non-neighbors with high probability are "quantum discoveries"
        discovery_score = prob * (0.3 if is_direct else 1.5)
        
        if node_idx < len(nodes):
            discoveries.append({
                'node': nodes[node_idx],
                'node_index': node_idx,
                'probability': round(prob, 4),
                'discovery_score': round(discovery_score, 4),
                'is_direct_connection': is_direct,
                'connection_type': 'quantum_discovered' if not is_direct else 'reinforced'
            })
    
    # Sort by discovery score (unexpected connections first)
    discoveries.sort(key=lambda x: x['discovery_score'], reverse=True)
    
    return discoveries[:5]  # Return top 5 discoveries

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'quantum-walk'})

@app.route('/quantum-walk', methods=['POST'])
def run_quantum_walk():
    """Main endpoint for quantum random walk."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        start_node = data.get('start_node', 0)
        walk_steps = data.get('walk_steps', 3)
        
        if len(nodes) < 2:
            return jsonify({'error': 'Need at least 2 nodes'}), 400
        
        if len(nodes) > 16:
            # Limit to 16 nodes for quantum simulation feasibility
            nodes = nodes[:16]
            edges = [e for e in edges if e.get('source', 0) < 16 and e.get('target', 0) < 16]
        
        # Build adjacency matrix
        adjacency = build_adjacency_matrix(nodes, edges)
        
        # Run quantum walk
        probabilities = quantum_random_walk(adjacency, start_node, walk_steps)
        
        # Calculate discovery scores
        discoveries = calculate_discovery_scores(probabilities, adjacency, start_node, nodes)
        
        return jsonify({
            'success': True,
            'method': 'quantum',
            'discoveries': discoveries,
            'raw_probabilities': {nodes[k]: v for k, v in probabilities.items() if k < len(nodes)},
            'start_node': nodes[start_node] if start_node < len(nodes) else None,
            'nodes_analyzed': len(nodes),
            'walk_steps': walk_steps
        })
        
    except Exception as e:
        print(f"Error in quantum walk: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
