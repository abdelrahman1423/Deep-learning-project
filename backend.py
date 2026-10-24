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

    # ===================== Load Cleaned Data =====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv_path = os.path.join(script_dir, "penguins.csv")
    ready_csv_path = os.path.join(script_dir, "Ready_penguins.csv")

    if not os.path.exists(ready_csv_path):
        from data_cleaning import prepare_penguin_data
        prepare_penguin_data(input_path=raw_csv_path, output_path=ready_csv_path)

    df = pd.read_csv(ready_csv_path)

    # Reconstruct 1D scaled OriginLocation from the one-hot columns in Ready_penguins.csv
    location_mapping = {"Torgersen": 0.0, "Biscoe": 0.5, "Dream": 1.0}
    if "OriginLocation" not in df.columns:
        df["OriginLocation"] = 0.0
        if "OriginLocation_Biscoe" in df.columns:
            df.loc[df["OriginLocation_Biscoe"] == 1, "OriginLocation"] = 0.5
        if "OriginLocation_Dream" in df.columns:
            df.loc[df["OriginLocation_Dream"] == 1, "OriginLocation"] = 1.0

    # Map class names to categorical codes generated by data_cleaning.py
    species_mapping = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
    if pd.api.types.is_numeric_dtype(df["Species"]):
        c1_code = species_mapping[class_1]
        c2_code = species_mapping[class_2]
    else:
        c1_code, c2_code = class_1, class_2

    # ===================== Filter Classes and Extract Features =====================
    df = df[df["Species"].isin([c1_code, c2_code])].reset_index(drop=True)

    # Features in Ready_penguins.csv are already min-max normalized to [0, 1]
    X = df[[feature_1, feature_2]].values.astype(float)
    y = np.array([1 if s == c1_code else -1 for s in df["Species"]])

    # ===================== Split Train / Test =====================
    class1_indices = np.where(y == 1)[0]
    class2_indices = np.where(y == -1)[0]

    np.random.shuffle(class1_indices)
    np.random.shuffle(class2_indices)

    train_indices = np.concatenate([class1_indices[:30], class2_indices[:30]])
    test_indices = np.concatenate([class1_indices[30:], class2_indices[30:]])

    # Shuffle training indices so classes are interleaved during gradient descent
    np.random.shuffle(train_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # ===================== Train Adaline =====================
    weights = np.random.uniform(-0.05, 0.05, size=2)
    bias = float(np.random.uniform(-0.05, 0.05)) if use_bias else 0.0

    final_epochs = 0
    mse = 0.0

    for epoch in range(epochs):
        final_epochs = epoch + 1

        for i in range(len(X_train)):
            net = weights[0] * X_train[i][0] + weights[1] * X_train[i][1]
            if use_bias:
                net = net + bias

            # Linear activation function: y = net
            prediction = net
            error = y_train[i] - prediction

            weights[0] = weights[0] + eta * error * X_train[i][0]
            weights[1] = weights[1] + eta * error * X_train[i][1]
            if use_bias:
                bias = bias + eta * error

        # Calculate Mean Squared Error (MSE) across the training set at the end of epoch
        total_error = 0.0
        for i in range(len(X_train)):
            net = weights[0] * X_train[i][0] + weights[1] * X_train[i][1]
            if use_bias:
                net = net + bias
            error = y_train[i] - net
            total_error += 0.5 * (error ** 2)

        mse = total_error / len(X_train)

        if mse <= mse_threshold:
            break

    # ===================== Test and Evaluate =====================
    predictions = []
    for i in range(len(X_test)):
        net = weights[0] * X_test[i][0] + weights[1] * X_test[i][1]
        if use_bias:
            net = net + bias

        # Signum activation function for testing
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

    b_val = bias if use_bias else 0.0
    if abs(weights[1]) > 0.0001:
        x_line = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200)
        y_line = -(weights[0] * x_line + b_val) / weights[1]
        ax.plot(x_line, y_line, "g-", linewidth=2, label="Decision Boundary")

    ax.scatter([], [], c="blue", marker="o", label=f"{class_1} (train)")
    ax.scatter([], [], c="red", marker="o", label=f"{class_2} (train)")
    ax.scatter([], [], c="blue", marker="x", label=f"{class_1} (test)")
    ax.scatter([], [], c="red", marker="x", label=f"{class_2} (test)")

    ax.set_xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    ax.set_ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title(f"Adaline: {class_1} vs {class_2}")
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
    def scale_sample_value(feat_name, val):
        if feat_name == "OriginLocation":
            return location_mapping.get(val, 0.0)
        val = float(val)
        # If user enters a raw unscaled value (> 1.0), normalize using penguins.csv min/max
        if val > 1.0 and os.path.exists(raw_csv_path):
            raw_df = pd.read_csv(raw_csv_path)
            col_s = pd.to_numeric(raw_df[feat_name], errors="coerce")
            min_v, max_v = col_s.min(), col_s.max()
            if max_v > min_v:
                return (val - min_v) / (max_v - min_v)
        return val

    sample_x1 = scale_sample_value(feature_1, sample_f1)
    sample_x2 = scale_sample_value(feature_2, sample_f2)

    net = weights[0] * sample_x1 + weights[1] * sample_x2
    if use_bias:
        net = net + bias

    if net >= 0:
        predicted_class = class_1
    else:
        predicted_class = class_2

    return {
        "predicted_class": predicted_class,
        "final_mse": round(float(mse), 6),
        "final_epochs": final_epochs,
        "accuracy": f"{accuracy:.2f}%",
        "confusion_matrix": {"TP": TP, "FN": FN, "FP": FP, "TN": TN},
        "weights": weights.tolist(),
        "bias": float(bias),
    }
