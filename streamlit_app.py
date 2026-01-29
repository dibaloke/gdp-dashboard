# Filename: interactive_regression_applet_logalpha.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Interactive Regression Explorer: OLS, Ridge, Lasso, ElasticNet (Log-scale alpha)")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Model Controls")

# Select multiple models for comparison
model_types = st.sidebar.multiselect(
    "Select Regression Models",
    ["OLS", "Ridge", "Lasso", "ElasticNet"],
    default=["OLS", "Ridge"]
)

fit_intercept = st.sidebar.checkbox("Fit Intercept", value=True)

# Log-scale alpha slider: user selects log10(alpha)
log_alpha = st.sidebar.slider(
    "Regularization strength alpha (log10 scale for Ridge/Lasso/ElasticNet)", 
    -3.0, 3.0, 0.0, 0.1
)
alpha = 10 ** log_alpha  # Convert log scale to actual alpha

l1_ratio = st.sidebar.slider("L1 ratio (ElasticNet only)", 0.0, 1.0, 0.5, 0.05)
noise_level = st.sidebar.slider("Noise level in data", 0.0, 5.0, 0.5, 0.1)

# ---------------------------
# Generate synthetic dataset
# ---------------------------
np.random.seed(42)
n_samples, n_features = 50, 5
feature_names = [f"x{i+1}" for i in range(n_features)]

# True coefficients
true_coef = np.array([1.5, -2.0, 0.0, 0.5, 3.0])

# Generate features
X = np.random.randn(n_samples, n_features)

# Generate response with controllable noise
y = X @ true_coef + np.random.randn(n_samples) * noise_level

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Interactive sliders for input features
# ---------------------------
st.sidebar.header("Input Feature Values")
input_features = []
for i, name in enumerate(feature_names):
    val = st.sidebar.slider(name, float(X[:, i].min()), float(X[:, i].max()), float(X[:, i].mean()))
    input_features.append(val)
input_features_scaled = scaler.transform([input_features])

# ---------------------------
# Fit models and display coefficients and equations
# ---------------------------
st.subheader("Regression Coefficients and Equations")
coef_df = pd.DataFrame({"Feature": feature_names})
equation_texts = []
pred_input_dict = {}

for mtype in model_types:
    if mtype == "OLS":
        model = LinearRegression(fit_intercept=fit_intercept)
    elif mtype == "Ridge":
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    elif mtype == "Lasso":
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
    elif mtype == "ElasticNet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=10000)
    
    model.fit(X_scaled, y)
    
    # Store coefficients
    coef_df[mtype] = model.coef_
    
    # Build regression equation
    equation = f"{mtype}: y = {model.intercept_:.3f}"
    for i, c in enumerate(model.coef_):
        sign = " + " if c >= 0 else " - "
        equation += f"{sign}{abs(c):.3f}*{feature_names[i]}"
    equation_texts.append(equation)
    
    # Prediction for user-defined input
    pred_input_dict[mtype] = model.predict(input_features_scaled)[0]

# Display coefficients
st.table(coef_df)

# Display equations
st.subheader("Regression Equations")
for eq in equation_texts:
    st.code(eq)

# ---------------------------
# Display prediction for input features
# ---------------------------
st.subheader("Prediction for Input Features")
st.write(pred_input_dict)

# ---------------------------
# Plot predictions vs true values
# ---------------------------
st.subheader("Predictions vs True Values")
pred_all_dict = {}
for mtype in model_types:
    if mtype == "OLS":
        model = LinearRegression(fit_intercept=fit_intercept)
    elif mtype == "Ridge":
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    elif mtype == "Lasso":
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
    elif mtype == "ElasticNet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=10000)
    
    model.fit(X_scaled, y)
    pred_all_dict[mtype] = model.predict(X_scaled)

df_plot = pd.DataFrame(pred_all_dict)
df_plot["True"] = y
st.line_chart(df_plot)

# ---------------------------
# Coefficient regularization path
# ---------------------------
if any(m in ["Ridge", "Lasso", "ElasticNet"] for m in model_types):
    st.subheader("Coefficient Regularization Path")
    alphas = np.logspace(-3, 3, 50)  # Match log scale slider
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mtype in model_types:
        if mtype == "OLS":
            continue
        coefs_path = []
        for a in alphas:
            if mtype == "Ridge":
                m = Ridge(alpha=a, fit_intercept=fit_intercept)
            elif mtype == "Lasso":
                m = Lasso(alpha=a, fit_intercept=fit_intercept, max_iter=10000)
            elif mtype == "ElasticNet":
                m = ElasticNet(alpha=a, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=10000)
            m.fit(X_scaled, y)
            coefs_path.append(m.coef_)
        coefs_path = np.array(coefs_path)
        for i in range(n_features):
            ax.plot(alphas, coefs_path[:, i], label=f"{mtype}-{feature_names[i]}")
    
    ax.set_xscale('log')
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Regularization Path")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# ---------------------------
# Real-time bar chart: True vs predicted coefficients
# ---------------------------
st.subheader("True vs Predicted Coefficients")
for mtype in model_types:
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    indices = np.arange(n_features)
    
    ax.bar(indices - width/2, true_coef, width, label="True", color="skyblue")
    ax.bar(indices + width/2, coef_df[mtype], width, label=mtype, color="orange")
    
    ax.set_xticks(indices)
    ax.set_xticklabels(feature_names)
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"True vs {mtype} Coefficients")
    ax.legend()
    
    st.pyplot(fig)
