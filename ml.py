import math
import os

def load_dataset(path):
    """
    Loads the dataset from a CSV file.
    
    The CSV is expected to have:
    - First column: instance ID (e.g., day) – ignored
    - Second column onward: features + target (last column)
    - First row: header row with column names
    
    Returns:
        dataset – list of lists containing the data rows (without the ID column)
        headers – list of feature names + target name
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file '{path}' not found.")
        
    dataset = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if parts:  # Skip empty lines
                dataset.append(parts[1:])  # Skip the day/ID column
    
    headers = dataset.pop(0)  # Remove and return the header row
    return dataset, headers


def get_entropy(rows):
    """
    Calculates the entropy of a dataset.
    
    Entropy measures impurity: 0 = pure, higher values = more mixed classes.
    
    Formula: H(S) = -Σ (p_i * log2(p_i)) for each class i
    """
    if not rows:
        return 0.0
    
    counts = {}
    for row in rows:
        label = row[-1]  # Target is always the last column
        counts[label] = counts.get(label, 0) + 1
            
    entropy = 0.0
    total = len(rows)
    
    for count in counts.values():
        p = count / total
        if p > 0:  # Avoid log(0)
            entropy -= p * math.log2(p)
        
    return entropy


def split_dataset(dataset, index):
    """
    Splits the dataset on a given feature index.
    
    Returns a dictionary: {feature_value: [rows without that feature]}
    """
    splits = {}

    for row in dataset:
        value = row[index]
        reduced_row = row[:index] + row[index+1:]  # Remove the split feature
        splits.setdefault(value, []).append(reduced_row)
        
    return splits


def find_best_split(rows):
    """
    Finds the feature that provides the highest information gain.
    
    Returns:
        best_index – index of the best feature
        best_gain  – corresponding information gain value
    """
    if not rows or len(rows[0]) <= 1:
        return -1, 0.0
        
    base_entropy = get_entropy(rows)
    best_gain = 0.0
    best_index = -1
    
    n_features = len(rows[0]) - 1  # Exclude target column

    for i in range(n_features):
        splits = split_dataset(rows, i)
        weighted_entropy = 0.0
        total_rows = len(rows)
        
        for subset in splits.values():
            weight = len(subset) / total_rows
            weighted_entropy += weight * get_entropy(subset)
            
        gain = base_entropy - weighted_entropy

        if gain > best_gain:
            best_gain = gain
            best_index = i
            
    return best_index, best_gain


def build_tree(rows, headers):
    """
    Recursively builds the decision tree using the ID3 algorithm.
    
    Stopping conditions:
        - All instances have the same label → leaf node with that label
        - No features left → leaf with majority class
        - No gain from any split → leaf with majority class
    """
    if not rows:
        return None
    
    # Count class labels
    counts = {}
    for row in rows:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
        
    # Pure node → return the label
    if len(counts) == 1:
        return list(counts.keys())[0]
        
    # No features left → return majority class
    if len(rows[0]) == 1:  # Only target remains
        return max(counts, key=counts.get)

    # Find best feature to split on
    best_idx, best_gain = find_best_split(rows)
    
    # No informative split → return majority class
    if best_gain == 0:
        return max(counts, key=counts.get)

    best_feature = headers[best_idx]
    tree = {best_feature: {}}
    
    # Remove the used feature from headers for recursion
    new_headers = headers[:best_idx] + headers[best_idx+1:]
    
    splits = split_dataset(rows, best_idx)
    
    # Build subtree for each branch
    for value, subset in splits.items():
        subtree = build_tree(subset, new_headers)
        tree[best_feature][value] = subtree
        
    return tree


def classify(observation, tree):
    """
    Classifies a new instance using the built decision tree.
    
    observation: dict with feature names as keys (case-sensitive)
    Returns the predicted class label or "Unknown" if path not found.
    """
    if not isinstance(tree, dict):
        return tree  # Leaf node
    
    feature = list(tree.keys())[0]
    value = observation.get(feature)
    
    if value not in tree[feature]:
        return "Unknown"  # Handle unseen feature values gracefully
    
    next_step = tree[feature][value]
    return classify(observation, next_step)


def run():
    """
    Main function to demonstrate the ID3 decision tree on the Play Tennis dataset.
    """
    filename = 'play_tennis.csv'
    dataset, headers = load_dataset(filename)

    print("Building decision tree...\n")
    my_tree = build_tree(dataset, headers)
    
    print("Decision Tree:")
    print(my_tree)

    print("\n--- Prediction Tests ---")
    
    test_cases = [
        {'outlook': 'Overcast', 'temp': 'Mild', 'humidity': 'Normal', 'wind': 'Weak'},
        {'outlook': 'Sunny', 'temp': 'Hot', 'humidity': 'High', 'wind': 'Weak'},
        {'outlook': 'Rain', 'temp': 'Cool', 'humidity': 'Normal', 'wind': 'Strong'}
    ]
    
    for case in test_cases:
        result = classify(case, my_tree)
        print(f"Input: {case}")
        print(f"Prediction: {result}\n")


# Run the demo when executed directly
if __name__ == "__main__":
    run()