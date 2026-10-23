import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os


def perceptron_backend(feature_1, feature_2, class_1, class_2, eta, epochs, mse_threshold, use_bias, sample_f1, sample_f2):

    # ===================== Load and Clean Data =====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "penguins.csv"))

    df["Species"] = df["Species"].fillna(df["Species"].mode()[0])
    df["OriginLocation"] = df["OriginLocation"].fillna(df["OriginLocation"].mode()[0])

    num_cols = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df.groupby("Species")[col].transform(lambda x: x.fillna(x.mean()))

    # ===================== Encode OriginLocation =====================
    location_mapping = {"Torgersen": 0, "Biscoe": 1, "Dream": 2}
    df["OriginLocation"] = df["OriginLocation"].map(location_mapping).astype(float)

    # ===================== Filter Classes and Extract Features =====================
    df = df[df["Species"].isin([class_1, class_2])].reset_index(drop=True)

    X = df[[feature_1, feature_2]].values.astype(float)
    y = np.array([1 if s == class_1 else -1 for s in df["Species"]])

    # ===================== Normalize Features =====================
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    X = (X - min_vals) / ranges

    # ===================== Split Train / Test =====================
    class1_indices = np.where(y == 1)[0]
    class2_indices = np.where(y == -1)[0]

    np.random.shuffle(class1_indices)
    np.random.shuffle(class2_indices)

    train_indices = np.concatenate([class1_indices[:30], class2_indices[:30]])
    test_indices = np.concatenate([class1_indices[30:], class2_indices[30:]])

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # ===================== Train Perceptron =====================
    weights = np.zeros(2)
    bias = 0.0

    for epoch in range(epochs):
        for i in range(len(X_train)):
            net = weights[0] * X_train[i][0] + weights[1] * X_train[i][1]
            if use_bias:
                net = net + bias

            if net >= 0:
                prediction = 1
            else:
                prediction = -1

            error = y_train[i] - prediction

            if error != 0:
                weights[0] = weights[0] + eta * error * X_train[i][0]
                weights[1] = weights[1] + eta * error * X_train[i][1]
                if use_bias:
                    bias = bias + eta * error

    # ===================== Test and Evaluate =====================
    predictions = []
    for i in range(len(X_test)):
        net = weights[0] * X_test[i][0] + weights[1] * X_test[i][1]
        if use_bias:
            net = net + bias

        if net >= 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    predictions = np.array(predictions)

    # ===================== Confusion Matrix =====================
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_test)):
        if y_test[i] == 1 and predictions[i] == 1:
            TP += 1
        elif y_test[i] == -1 and predictions[i] == -1:
            TN += 1
        elif y_test[i] == -1 and predictions[i] == 1:
            FP += 1
        elif y_test[i] == 1 and predictions[i] == -1:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100

    # ===================== Plot Decision Boundary =====================
    fig, ax = plt.subplots()

    for i in range(len(X_train)):
        if y_train[i] == 1:
            ax.scatter(X_train[i][0], X_train[i][1], c="blue", marker="o")
        else:
            ax.scatter(X_train[i][0], X_train[i][1], c="red", marker="o")

    for i in range(len(X_test)):
        if y_test[i] == 1:
            ax.scatter(X_test[i][0], X_test[i][1], c="blue", marker="x")
        else:
            ax.scatter(X_test[i][0], X_test[i][1], c="red", marker="x")

    if abs(weights[1]) > 0.0001:
        x_line = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200)
        y_line = -(weights[0] * x_line + bias) / weights[1]
        ax.plot(x_line, y_line, "g-", linewidth=2, label="Decision Boundary")

    ax.scatter([], [], c="blue", marker="o", label=f"{class_1} (train)")
    ax.scatter([], [], c="red", marker="o", label=f"{class_2} (train)")
    ax.scatter([], [], c="blue", marker="x", label=f"{class_1} (test)")
    ax.scatter([], [], c="red", marker="x", label=f"{class_2} (test)")

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title(f"Perceptron: {class_1} vs {class_2}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close()

    # ===================== Display Results =====================
    st.subheader("Confusion Matrix")
    cm_table = pd.DataFrame(
        [[TP, FN], [FP, TN]],
        index=[f"Actual {class_1}", f"Actual {class_2}"],
        columns=[f"Predicted {class_1}", f"Predicted {class_2}"],
    )
    st.table(cm_table)
    st.write(f"**Overall Accuracy: {accuracy:.2f}%**")

    # ===================== Classify New Sample =====================
    if feature_1 == "OriginLocation":
        sample_f1 = location_mapping.get(sample_f1, 0)
    if feature_2 == "OriginLocation":
        sample_f2 = location_mapping.get(sample_f2, 0)

    sample_x1 = (float(sample_f1) - min_vals[0]) / ranges[0]
    sample_x2 = (float(sample_f2) - min_vals[1]) / ranges[1]

    net = weights[0] * sample_x1 + weights[1] * sample_x2
    if use_bias:
        net = net + bias

    if net >= 0:
        predicted_class = class_1
    else:
        predicted_class = class_2

    return {
        "predicted_class": predicted_class,
        "final_epochs": epochs,
        "accuracy": f"{accuracy:.2f}%",
        "confusion_matrix": {"TP": TP, "FN": FN, "FP": FP, "TN": TN},
        "weights": weights.tolist(),
        "bias": float(bias),
    }


def adaline_backend(feature_1, feature_2, class_1, class_2, eta, epochs, mse_threshold, use_bias, sample_f1, sample_f2):
    pass
