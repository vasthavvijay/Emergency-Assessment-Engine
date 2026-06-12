from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Prior Probabilities for Root Nodes
PRIORS = {
    'incident': {'medical': 0.40, 'fire': 0.15, 'accident': 0.30, 'other': 0.15},
    'severity': {'High': 0.35, 'Low': 0.65},
    'location': {'High': 0.25, 'Low': 0.75}
}

# Conditional Probability Table (CPT) for P(Critical = True | Incident, Severity, Location)
CPT_CRITICAL_TRUE = {
    ('fire', 'High', 'High'): 0.98,
    ('fire', 'High', 'Low'): 0.85,
    ('fire', 'Low', 'High'): 0.70,
    ('fire', 'Low', 'Low'): 0.45,
    
    ('medical', 'High', 'High'): 0.92,
    ('medical', 'High', 'Low'): 0.80,
    ('medical', 'Low', 'High'): 0.60,
    ('medical', 'Low', 'Low'): 0.30,
    
    ('accident', 'High', 'High'): 0.88,
    ('accident', 'High', 'Low'): 0.75,
    ('accident', 'Low', 'High'): 0.50,
    ('accident', 'Low', 'Low'): 0.20,
    
    ('other', 'High', 'High'): 0.60,
    ('other', 'High', 'Low'): 0.40,
    ('other', 'Low', 'High'): 0.30,
    ('other', 'Low', 'Low'): 0.10,
}

def discretize_score(score_str):
    """Maps raw form inputs into discrete states or preserves unknown flag."""
    if score_str == 'unknown' or score_str is None:
        return 'unknown'
    try:
        score = float(score_str)
        return 'High' if score > 5 else 'Low'
    except ValueError:
        return 'unknown'

def bayesian_inference(evidence):
    """
    Performs exact inference by marginalizing over unobserved variables
    using the joint probability factorization rule.
    """
    states_I = [evidence['incident']] if evidence['incident'] != 'unknown' else list(PRIORS['incident'].keys())
    states_S = [evidence['severity']] if evidence['severity'] != 'unknown' else list(PRIORS['severity'].keys())
    states_L = [evidence['location']] if evidence['location'] != 'unknown' else list(PRIORS['location'].keys())
    
    joint_critical_true = 0.0
    joint_critical_false = 0.0
    
    # Algorithmic summation over the state space
    for i in states_I:
        for s in states_S:
            for l in states_L:
                # Calculate probability product of independent root nodes
                p_parents = PRIORS['incident'][i] * PRIORS['severity'][s] * PRIORS['location'][l]
                
                # Retrieve conditional factor
                p_c_given_parents = CPT_CRITICAL_TRUE.get((i, s, l), 0.50)
                
                # Accumulate values into joint probability spaces
                joint_critical_true += p_c_given_parents * p_parents
                joint_critical_false += (1.0 - p_c_given_parents) * p_parents
                
    # Normalize values using Bayes' Theorem to find posterior probability
    total_evidence_weight = joint_critical_true + joint_critical_false
    if total_evidence_weight == 0:
        return 0.50
        
    return joint_critical_true / total_evidence_weight

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    
    evidence = {
        'incident': data.get('incident_type', 'other'),
        'severity': discretize_score(data.get('severity_score')),
        'location': discretize_score(data.get('location_risk'))
    }
    
    prob = bayesian_inference(evidence)
    
    return jsonify({
        'emergency_probability': round(prob * 100, 2),
        'status': 'critical' if prob >= 0.75 else 'standard'
    })

if __name__ == '__main__':
    app.run(debug=True)hello