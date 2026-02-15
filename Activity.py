# use any human activity dataset for this ....
# ===============================
# Step 0: Import Libraries
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import cv2
print("libraries imported::")

# ===============================
# Step 1: Load and Clean Dataset
# ===============================
data = pd.read_csv(
    "WISDM_ar_v1.1_raw.txt",
    header=None,
    names=["user", "activity", "timestamp", "x", "y", "z"],
    sep=",",
    on_bad_lines='skip'  # <-- THIS IGNORES CORRUPTED LINES
   
)

# Remove rows with missing data & clean ';' in z column
data.dropna(inplace=True)
for col in ["x", "y", "z"]:
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(";", "", regex=False), errors="coerce")
data.dropna(inplace=True)

print("Data loaded and cleaned ✅")

# ===============================
# Step 2: Prepare Features & Labels
# ===============================
X = data[["x", "y", "z"]]
y = data["activity"]

# Encode activities as numbers
activity_map = {label: idx for idx, label in enumerate(y.unique())}
activity_map_inv = {v: k for k, v in activity_map.items()}
y_encoded = y.map(activity_map)

# ===============================
# Step 3: Split Data into Train & Test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("Data split into training and testing ✅")

# ===============================
# Step 4: Train Random Forest Classifier
# ===============================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained. Test Accuracy: {accuracy:.4f}")

# =====================================================
# Step 5: Real-Time Activity Simulation with Live Graph
# =====================================================
window_width, window_height = 600, 400
cv2.namedWindow("HAR Live Simulation")

# Initialize blank frame for plotting
frame = np.ones((window_height, window_width, 3), dtype=np.uint8) * 255

# Colors for x, y, z lines
colors = {"x": (255, 0, 0), "y": (0, 255, 0), "z": (0, 0, 255)}

# Buffers to store last 50 readings for live plotting
buffer_size = 50
x_vals, y_vals, z_vals = [], [], []


for i in range(100):  # simulate 100 time steps
    idx = random.randint(0, len(X_test) - 1)
    sample = X_test.iloc[[idx]]  # DataFrame (correct)

    pred_encoded = rf_model.predict(sample)[0]
    predicted_activity = activity_map_inv[pred_encoded]

    # ✅ Correct value extraction
    x_vals.append(sample.iloc[0]["x"])
    y_vals.append(sample.iloc[0]["y"])
    z_vals.append(sample.iloc[0]["z"])

    if len(x_vals) > buffer_size:
        x_vals.pop(0)
        y_vals.pop(0)
        z_vals.pop(0)

    frame[:] = 255

    cv2.putText(frame, f"Predicted Activity: {predicted_activity}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    margin = 50
    scale = 10
    for j in range(1, len(x_vals)):
        cv2.line(frame,
                 (margin + (j-1)*10, int(window_height/2 - x_vals[j-1]*scale)),
                 (margin + j*10, int(window_height/2 - x_vals[j]*scale)),
                 (255, 0, 0), 2)

        cv2.line(frame,
                 (margin + (j-1)*10, int(window_height/2 - y_vals[j-1]*scale)),
                 (margin + j*10, int(window_height/2 - y_vals[j]*scale)),
                 (0, 255, 0), 2)

        cv2.line(frame,
                 (margin + (j-1)*10, int(window_height/2 - z_vals[j-1]*scale)),
                 (margin + j*10, int(window_height/2 - z_vals[j]*scale)),
                 (0, 0, 255), 2)

    cv2.imshow("HAR Live Simulation", frame)

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Simulation complete ✅")
