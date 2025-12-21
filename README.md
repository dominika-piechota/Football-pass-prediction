# Football Pass Receiver Prediction ⚽🏃‍♂️

This project aims to predict the most likely receiver of a football pass based on a snapshot of player positions (coordinates) at the moment the pass is initiated. The solution was developed as part of the *Introduction to Machine Learning* course competition.

## 📌 Project Overview

Predicting a pass receiver is a complex task that goes beyond simple distance calculations. In professional football, decisions are relative—a player is chosen because they are a **better option** compared to others, considering tactical positioning, opponent pressure, and passing lanes.

To capture this, we moved away from simple classification and treated the problem as **Learning to Rank (LTR)**. Each of the 22 players on the pitch is ranked, and the model identifies who is most likely to receive the ball.

**Final Performance:**

* **Top-1 Accuracy:** 42.15% (Validation Set)
* **Brier Score:** 0.712

## 🚀 Technical Stack

* **Languages:** Python
* **Machine Learning:** * **LightGBM & XGBoost** (Ensemble of Rankers)
* **Scikit-learn** (Preprocessing & Evaluation)


* **Data Manipulation:** Pandas, NumPy
* **Mathematical Modeling:** Geometric computations for interception risk.

## 📐 Key Innovation: The Vertex-Focus Ellipse

The core of our feature engineering is a dynamic **geometric interception model**. Instead of checking a simple radius around players, we calculate an ellipse where:

1. The **Sender** is at one vertex.
2. The **Potential Receiver** is at the focus point.

### Tactical Rationale:

* **Cone of Uncertainty:** The shape is naturally narrow near the sender and wider near the receiver, accurately modeling how a defender's ability to intercept increases as the ball travels.
* **Dynamic Scaling:** The ellipse area scales quadratically with pass distance, reflecting the higher risk of long-range passes.
* **Interceptor Detection:** We transform global coordinates to the ellipse's local coordinate system to efficiently count "potential interceptors" (opponents inside the shape).

## 📊 Feature Highlights

The model relies on several high-impact features derived from raw coordinates:

* **`opponents_in_ellipse`**: The count of opponents near the passing lane (calculated via the Vertex-Focus model).
* **`receiver_closest_opponent_dist`**: Measures immediate marking pressure on the target.
* **`receiver_closest_3_teammates_dist`**: Captures local passing options and support.
* **`direction`**: A weighted feature penalizing backward passes and rewarding progressive play.
* **`pitch_zones`**: Contextual data based on whether the pass occurs in the defensive, midfield, or attacking third.

## 🏗️ Model Architecture

The final system is an **ensemble (50:50 blend)** of two state-of-the-art gradient boosting frameworks:

1. **LightGBM Ranker:** Optimized using `lambdarank`. It acts as the "Physics Specialist," excelling at capturing deep interactions between geometric features.
2. **XGBoost Ranker:** Optimized using `rank:ndcg`. It acts as the "Tactical Specialist," providing robust generalizations based on player density and pressure statistics.

Final probabilities are refined using a **Softmax function with a Temperature parameter**, tuned via Random Search to optimize prediction confidence.

---

### How to run

1. Clone the repository.
2. Ensure you have the datasets (not provided here): `input_train_set.csv`, `output_train_set.csv`, and `input_test_set.csv`.
3. Run the main script:
```bash
python main.py

```



**Authors:** Dominika Piechota, Paweł Dorosz
