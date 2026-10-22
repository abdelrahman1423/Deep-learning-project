import streamlit as st
import os

# Import the backend function headers from your teammate's backend.py file
try:
    from backend import perceptron_backend, adaline_backend
except ImportError:
    # Fallback placeholder definitions in case backend.py doesn't exist yet
    def perceptron_backend(feature_1, feature_2, class_1, class_2, eta, epochs, mse_threshold, use_bias, sample_f1, sample_f2):
        pass
    def adaline_backend(feature_1, feature_2, class_1, class_2, eta, epochs, mse_threshold, use_bias, sample_f1, sample_f2):
        pass

# ==============================================================================
# STREAMLIT FRONTEND GUI
# ==============================================================================

st.set_page_config(page_title="Classification GUI", layout="wide")
st.title("🐧 Perceptron & Adaline Classification Interface")

# --- Specific Dataset Options ---
FEATURE_OPTIONS = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]

CLASS_NAMES = ["Adelie", "Chinstrap", "Gentoo"]
CLASS_PAIRS = {
    "Adelie & Chinstrap": ("Adelie", "Chinstrap"),
    "Adelie & Gentoo": ("Adelie", "Gentoo"),
    "Chinstrap & Gentoo": ("Chinstrap", "Gentoo")
}

# Mapping class names to their image filenames
SPECIE_IMAGES = {
    "Adelie": "Adelie.png",
    "Chinstrap": "Chinstrap.png",
    "Gentoo": "Gentoo.png"
}

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Configuration & Inputs")

# 1. Feature Selection (User selects exactly two)
selected_features = st.sidebar.multiselect(
    "Select TWO features:",
    options=FEATURE_OPTIONS,
    default=FEATURE_OPTIONS[:2]
)

# 2. Class Pair Selection
selected_pair = st.sidebar.selectbox("Select two classes:", list(CLASS_PAIRS.keys()))
class_1, class_2 = CLASS_PAIRS[selected_pair]

# 3. Hyperparameters
eta = st.sidebar.number_input("Learning Rate (eta):", min_value=0.0001, max_value=1.0, value=0.01, step=0.005, format="%.4f")
epochs = st.sidebar.number_input("Number of Epochs (m):", min_value=1, max_value=10000, value=100, step=10)
mse_threshold = st.sidebar.number_input("MSE Threshold:", min_value=0.0, max_value=1.0, value=0.01, step=0.005, format="%.4f")

# 4. Bias Checkbox
use_bias = st.sidebar.checkbox("Add Bias", value=True)

# 5. Algorithm Selection
algorithm = st.sidebar.radio("Choose Algorithm:", ("Perceptron", "Adaline"))

# --- Main Panel: Sample Input ---
st.header("🔍 Classification Input Sample")

if len(selected_features) != 2:
    st.warning("⚠️ Please select exactly **TWO** features from the sidebar to proceed.")
    st.stop()

# Helper function to render correct input widget based on feature type
def render_feature_input(feature_name, key_suffix):
    if feature_name == "OriginLocation":
        return st.selectbox(
            f"Select value for '{feature_name}':",
            options=["Biscoe", "Dream", "Torgersen"],
            key=f"input_{key_suffix}"
        )
    else:
        return st.number_input(
            f"Enter value for '{feature_name}':",
            value=0.0,
            key=f"input_{key_suffix}"
        )

col_f1, col_f2 = st.columns(2)
with col_f1:
    sample_f1 = render_feature_input(selected_features[0], "f1")
with col_f2:
    sample_f2 = render_feature_input(selected_features[1], "f2")

run_button = st.button("🚀 Train & Classify Sample", type="primary")

# --- Execution & Dispatch to Backend ---
if run_button:
    f1_name = selected_features[0]
    f2_name = selected_features[1]

    # Send values to the appropriate backend function header
    if algorithm == "Perceptron":
        result = perceptron_backend(
            feature_1=f1_name,
            feature_2=f2_name,
            class_1=class_1,
            class_2=class_2,
            eta=eta,
            epochs=int(epochs),
            mse_threshold=mse_threshold,
            use_bias=use_bias,
            sample_f1=sample_f1,
            sample_f2=sample_f2
        )
    else:  # Adaline
        result = adaline_backend(
            feature_1=f1_name,
            feature_2=f2_name,
            class_1=class_1,
            class_2=class_2,
            eta=eta,
            epochs=int(epochs),
            mse_threshold=mse_threshold,
            use_bias=use_bias,
            sample_f1=sample_f1,
            sample_f2=sample_f2
        )

    # Display Results Output with Specific Image
    st.markdown("---")
    st.subheader("📊 Output Results")
    
    if result is not None and isinstance(result, dict) and "predicted_class" in result:
        pred_specie = result["predicted_class"]
        
        if pred_specie in CLASS_NAMES:
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.success(f"**Predicted Species:** `{pred_specie}`")
                
                if "final_mse" in result:
                    st.write(f"Final MSE: `{result['final_mse']}`")
                if "final_epochs" in result:
                    st.write(f"Final Epochs: `{result['final_epochs']}`")
                
                with st.expander("Show detailed backend output"):
                    st.json(result)
            
            with res_col2:
                image_filename = SPECIE_IMAGES[pred_specie]
                
                if os.path.exists(image_filename):
                    st.image(image_filename, caption=f"Representative photo of {pred_specie} penguin.", use_container_width=True)
                else:
                    st.error(f"Image not found. Ensure **`{image_filename}`** exists in the script directory.")
        else:
            st.error(f"Error: Backend returned an invalid or unrecognized species: `{pred_specie}`.")
            st.write("Expected values are: 'Adelie', 'Chinstrap', or 'Gentoo'.")
            st.json(result)
    else:
        st.info("The backend function ran, but no output dictionary containing 'predicted_class' was returned yet.")