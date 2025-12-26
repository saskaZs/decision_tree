# 🌳 ID3 Decision Tree Classifier from Scratch

A clean implementation of a **Decision Tree Classifier** using the **ID3 algorithm**, built entirely from scratch in pure Python — no external ML libraries like scikit-learn.

This version works directly with lists and dictionaries, demonstrating every step of the algorithm (entropy, information gain, recursive splitting) in a transparent and easy-to-understand way.

---

## 🚀 Features

- **Pure Python** – No dependencies beyond the standard library  
- **ID3 Algorithm** – Greedy selection of the attribute with the highest information gain  
- Handles only **categorical/nominal features** (classic ID3 limitation)  
- Prints the decision tree as a readable nested dictionary  
- Includes a simple classification function for new observations  
- Tested on the classic "Play Tennis" dataset  

---

## 📂 Project Structure
`ml.py`               # Main implementation (Decision Tree logic)
`play_tennis.csv`     # Classic "Play Tennis" dataset
`README.md`           # This file


---

## 📦 Dataset – Play Tennis

The famous 14-instance dataset that decides whether to play tennis based on weather conditions.

| Feature    | Possible Values                  |
|------------|----------------------------------|
| outlook    | Sunny, Overcast, Rain            |
| temp       | Hot, Mild, Cool                  |
| humidity   | High, Normal                     |
| wind       | Weak, Strong                     |
| **play**   | Yes, No                          |

The CSV file includes a header row and a "day" column (D1–D14) which is ignored during training.

---

## 🧠 Theoretical Background

The ID3 algorithm builds a decision tree by repeatedly splitting the dataset on the attribute that provides the greatest reduction in uncertainty.

### 1. Entropy

Entropy quantifies the impurity or randomness in a dataset.

$$ H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i) $$

- $S$: current dataset
- $c$: number of distinct class labels
- $p_i$: proportion of instances belonging to class $i$

Entropy is 0 when all instances belong to the same class (pure node) and maximum (1 for binary classes) when classes are perfectly balanced.

### 2. Information Gain

Information Gain measures how much entropy is reduced by splitting on a particular attribute $A$:

$$ IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot H(S_v) $$

- $H(S)$: entropy of the parent dataset
- $S_v$: subset of instances where attribute $A$ has value $v$
- The second term is the **weighted average entropy** of the child nodes

At each step, ID3 selects the attribute with the **highest Information Gain** as the splitting node.

The algorithm stops when:
- All instances have the same label → return that label (leaf)
- No attributes remain → return the majority class
- No gain from any split → return the majority class

---

## 🛠️ How to Run

1. Make sure both files (`ml.py` and `play_tennis.csv`) are in the same directory.

2. Run the script:

```bash
python ml.py
```

3. Expected output:
```
Decision Tree:
{'outlook': {'Sunny': {'humidity': {'High': 'No', 'Normal': 'Yes'}}, 
             'Overcast': 'Yes', 
             'Rain': {'wind': {'Weak': 'Yes', 'Strong': 'No'}}}}

--- Prediction Tests ---
Input: {'outlook': 'Overcast', 'temp': 'Mild', 'humidity': 'Normal', 'wind': 'Weak'}
Prediction: Yes

Input: {'outlook': 'Sunny', 'temp': 'Hot', 'humidity': 'High', 'wind': 'Weak'}
Prediction: No

Input: {'outlook': 'Rain', 'temp': 'Cool', 'humidity': 'Normal', 'wind': 'Strong'}
Prediction: No
```
