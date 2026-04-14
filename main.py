# quiet the tensorflow spam in the terminal
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def evaluate_model(y_test, model_pred, class_names, model_name):
    print(f"\n--- {model_name} Results ---")
    print(classification_report(y_test, model_pred, target_names=class_names, digits=4))
    
    cm = confusion_matrix(y_test, model_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

def main():
    print("[Intrusionator] Loading the datasets 'train.csv', 'val.csv', and 'test.csv'...")
    try:
        train_df = pd.read_csv("train.csv")
        val_df = pd.read_csv("val.csv")
        test_df = pd.read_csv("test.csv")
    except FileNotFoundError:
        print("Error: Datasets not found. Please ensure they are in the same directory.")
        return

    # separate the answers (Label) from the network data (Features)
    X_train = train_df.drop("Label", axis=1)
    y_train = train_df["Label"]
    
    X_val = val_df.drop("Label", axis=1)
    y_val = val_df["Label"]
    
    X_test = test_df.drop("Label", axis=1)
    y_test = test_df["Label"]

    print("[Intrusionator] Preprocessing and encoding the dataset...\n")
    
    # 1. Target Encoding (Y)
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_val_encoded = le_target.transform(y_val)
    y_test_encoded = le_target.transform(y_test)
    class_names = le_target.classes_
    num_classes = len(class_names)

    # 2. Feature Encoding (X) - One-Hot Encoding
    X_train_numeric = pd.get_dummies(X_train, columns=["Switch ID", "Port Number"], drop_first=True)
    X_val_numeric = pd.get_dummies(X_val, columns=["Switch ID", "Port Number"], drop_first=True)
    X_test_numeric = pd.get_dummies(X_test, columns=["Switch ID", "Port Number"], drop_first=True)
    
    # Ensure validation and test sets have the exact same columns as train
    X_val_numeric = X_val_numeric.reindex(columns=X_train_numeric.columns, fill_value=0)
    X_test_numeric = X_test_numeric.reindex(columns=X_train_numeric.columns, fill_value=0)

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_val_scaled = scaler.transform(X_val_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val_numeric.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns)

    # --- GRAPH 1: PCA Scatter Plot ---
    print("[Intrusionator - Preprocessing] Generating PCA Scatter Plot...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train_scaled)
    
    plt.figure(figsize=(10, 8))
    hue_labels = [class_names[i] for i in y_train_encoded]
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hue_labels, palette="tab10", s=15, alpha=0.7)
    plt.title("PCA of UNR-IDD Network Traffic (Pre-training Sanity Check)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Traffic Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # MACHINE LEARNING - Decision Tree
    print("\n[Intrusionator - ML] Running the Decision Tree ML Model (Week 6)...")
    start_time = time.time()
    
    # Parameters: Using Gini, restricted depth to prevent overfitting
    dt_model = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=5, random_state=42)
    dt_model.fit(X_train_scaled, y_train_encoded)
    dt_predictions = dt_model.predict(X_test_scaled)
    
    ml_time = time.time() - start_time
    print(f"            -> Custom Parameters: criterion=\"gini\", max_depth=10")
    print(f"            -> Training & Prediction completed in {ml_time:.2f} seconds.")
    
    evaluate_model(y_test_encoded, dt_predictions, class_names, "Decision Tree (ML)")
    
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] 
    plt.figure(figsize=(12, 6))
    plt.title("Decision Tree - Top 15 Feature Importances")
    plt.bar(range(15), importances[indices], align="center", color="green")
    plt.xticks(range(15), [X_train_scaled.columns[i] for i in indices], rotation=90)
    plt.xlim([-1, 15])
    plt.tight_layout()
    plt.show()

    # DEEP LEARNING - Neural Network
    print("\n[Intrusionator - DL] Running the Neural Network DL Model (Week 7)...")
    start_time = time.time()
    
    # Parameters: k=15 instead of 11
    print("            -> Applying ANOVA Feature Selection (k=15)...")
    selector = SelectKBest(score_func=f_classif, k=15)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_encoded)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    # Get the names of the selected features
    selected_mask = selector.get_support()
    selected_feature_names = X_train_scaled.columns[selected_mask].tolist()
    print("            -> Features chosen by ANOVA:")
    for feature in selected_feature_names:
        print(f"                 - {feature}")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    # Parameters: 3 layers of 64 (wider/shallower), ReLU, Dropout 0.3
    dl_model = Sequential()
    dl_model.add(Input(shape=(X_train_selected.shape[1],)))
    
    for _ in range(3): 
        dl_model.add(Dense(64))
        dl_model.add(ReLU())
        dl_model.add(Dropout(0.3))
        
    dl_model.add(Dense(num_classes, activation="softmax"))
    
    # Parameters: Standard Adam with higher learning rate
    optimizer = Adam(learning_rate=0.005)
    dl_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Parameters: batch_size=128
    history_data = dl_model.fit(X_train_selected, y_train_encoded, 
                                batch_size=128, epochs=100, 
                                validation_data=(X_val_selected, y_val_encoded), 
                                callbacks=[early_stop], 
                                verbose=0)
    
    actual_epochs = len(history_data.history["loss"])
    dl_time = time.time() - start_time
    print(f"            -> Training automatically stopped at Epoch {actual_epochs} of 100 to prevent overfitting!")
    print(f"            -> Training & Prediction completed in {dl_time:.2f} seconds.")
    
    dl_probs = dl_model.predict(X_test_selected, verbose=0)
    dl_predictions = np.argmax(dl_probs, axis=1)
    
    evaluate_model(y_test_encoded, dl_predictions, class_names, "Neural Network (DL)")

    # --- GRAPH 4: Deep Learning Training History ---
    print("[Intrusionator - DL] Generating Training History Graph...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_data.history["accuracy"], label="Training Accuracy", color="blue")
    plt.plot(history_data.history["val_accuracy"], label="Validation Accuracy", color="orange")
    plt.title("DL Model Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_data.history["loss"], label="Training Loss", color="blue")
    plt.plot(history_data.history["val_loss"], label="Validation Loss", color="orange")
    plt.title("DL Model Loss (Catching Overfitting)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("[Intrusionator] Done running all models and generating graphs!")

if __name__ == "__main__":
    main()