import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Neural Network Learning Demo",
    page_icon="🧠",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------
st.title("🧠 Interactive Neural Network Learning Demo")
st.write(
    "This demo shows how a neural network learns the XOR problem "
    "using TensorFlow."
)


# -----------------------------
# XOR Dataset
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)


# -----------------------------
# User controls
# -----------------------------
st.sidebar.header("⚙️ Training Settings")

epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01
)


# -----------------------------
# Create Neural Network
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# -----------------------------
# Compile model
# -----------------------------
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate
)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# Train model
# -----------------------------
if st.button("🚀 Train Neural Network"):

    with st.spinner("Training the neural network..."):

        history = model.fit(
            X,
            y,
            epochs=epochs,
            verbose=0
        )

    st.success("Training completed!")


    # -----------------------------
    # Display accuracy
    # -----------------------------
    final_accuracy = history.history["accuracy"][-1]
    final_loss = history.history["loss"][-1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Training Accuracy",
            f"{final_accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Final Loss",
            f"{final_loss:.4f}"
        )


    # -----------------------------
    # Loss Graph
    # -----------------------------
    st.subheader("📉 Training Loss")

    fig, ax = plt.subplots()

    ax.plot(history.history["loss"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss During Training")

    st.pyplot(fig)


    # -----------------------------
    # Accuracy Graph
    # -----------------------------
    st.subheader("📈 Training Accuracy")

    fig2, ax2 = plt.subplots()

    ax2.plot(history.history["accuracy"])

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy During Training")

    st.pyplot(fig2)


    # -----------------------------
    # Predictions
    # -----------------------------
    st.subheader("🔍 Model Predictions")

    predictions = model.predict(X, verbose=0)

    for i in range(len(X)):

        prediction = predictions[i][0]

        predicted_class = 1 if prediction >= 0.5 else 0

        st.write(
            f"Input: {X[i].astype(int)} → "
            f"Prediction: {predicted_class} "
            f"(Probability: {prediction:.4f})"
        )


# -----------------------------
# Explanation
# -----------------------------
st.divider()

st.subheader("📚 How does this neural network work?")

st.write("""
The neural network receives two input values (X1 and X2).

The hidden layers process these values using neurons and activation
functions.

During training, TensorFlow adjusts the weights of the neurons so that
the network learns the XOR relationship.

Finally, the output layer predicts either Class 0 or Class 1.
""")