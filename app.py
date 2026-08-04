import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Machine Learning Internship Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.main{
    background-color:#F5F7FA;
}

/* Main container */
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

/* Header Card */
.header-card{
    background: linear-gradient(90deg,#0F4C81,#1976D2);
    color:white;
    padding:22px;
    border-radius:12px;
    text-align:center;
    margin-bottom:20px;
}

.header-card h1{
    margin:0;
    font-size:36px;
}

.header-card p{
    margin-top:8px;
    font-size:16px;
}

/* Metric Cards */
.metric-card{
    background:white;
    padding:18px;
    border-radius:10px;
    text-align:center;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

/* Result Card */
.result-card{
    background:#EAF4FF;
    padding:20px;
    border-radius:10px;
    border-left:6px solid #1976D2;
}

/* Buttons */
div[data-testid="stButton"] > button{
    width:100%;
    background:#1565C0;
    color:white;
    border:none;
    border-radius:8px;
    height:46px;
    font-weight:600;
}

div[data-testid="stButton"] > button:hover{
    background:#0D47A1;
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#F8F9FB;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

house_model = joblib.load("models/house_model.pkl")

kmeans_model = joblib.load("models/kmeans_model.pkl")
kmeans_scaler = joblib.load("models/scalar.pkl")

svm_model = joblib.load("models/svm_cat_dog_classifier.pkl")
svm_scaler = joblib.load("models/svm_scaler.pkl")

gesture_model = load_model("models/leap_gesture_recognition_model.h5")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="header-card">
<h1>Machine Learning Internship Dashboard</h1>
<p>Interactive demonstration of four machine learning projects</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD CARDS
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Projects", "4")

with c2:
    st.metric("Models", "4")

with c3:
    st.metric("Algorithms", "4")

with c4:
    st.metric("Status", "Ready")

st.write("")

project = st.radio(
    "",
    [
        "House Price Prediction",
        "Customer Segmentation",
        "Cat vs Dog Classification",
        "Hand Gesture Recognition"
    ],
    horizontal=True
)

st.divider()


# =====================================================
# HOUSE PRICE PREDICTION
# =====================================================

if project == "House Price Prediction":

    st.header("House Price Prediction")
    st.write("Predict the estimated selling price of a house using a trained Linear Regression model.")

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Input Features")

        area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=10000,
            value=1500
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

        predict = st.button("Predict House Price")

    with right:

        st.subheader("Prediction")

        if predict:

            input_data = pd.DataFrame({
                "GrLivArea": [area],
                "BedroomAbvGr": [bedrooms],
                "FullBath": [bathrooms]
            })

            prediction = house_model.predict(input_data)

            st.markdown(f"""
            <div class="result-card">
                <h3>Estimated House Price</h3>
                <h2>${prediction[0]:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.info("Enter the house details and click **Predict House Price**.")

    st.write("")

    with st.expander("Model Information"):

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Algorithm**")
            st.write("Linear Regression")

            st.write("**Features Used**")
            st.write("""
- Living Area
- Bedrooms
- Bathrooms
""")

        with col2:
            st.write("**Dataset**")
            st.write("Ames Housing Dataset")

            st.write("**Purpose**")
            st.write("Estimate the selling price of a house based on selected features.")

# =====================================================
# CUSTOMER SEGMENTATION
# =====================================================

elif project == "Customer Segmentation":

    st.header("Customer Segmentation")
    st.write("Predict the customer segment using the trained K-Means clustering model.")

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Customer Details")

        recency = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=30
        )

        frequency = st.number_input(
            "Frequency",
            min_value=1,
            value=5
        )

        monetary = st.number_input(
            "Monetary Value (£)",
            min_value=0.0,
            value=500.0
        )

        predict = st.button("Find Customer Segment")

    with right:

        st.subheader("Prediction")

        if predict:

            customer = pd.DataFrame({
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary]
            })

            # Scale input
            customer_scaled = kmeans_scaler.transform(customer)

            # Predict cluster
            cluster = kmeans_model.predict(customer_scaled)[0]

            cluster_names = {
                0: "Occasional Customers",
                1: "Regular Customers",
                2: "At-Risk Customers",
                3: "VIP Customers"
            }

            st.markdown(f"""
            <div class="result-card">
                <h3>Customer Segment</h3>
                <h2>{cluster_names[cluster]}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"**Cluster Number:** {cluster}")

            if cluster == 3:
                st.success(
                    "These customers purchase frequently and spend the most. They are your most valuable customers."
                )

            elif cluster == 2:
                st.warning(
                    "These customers are at risk of leaving. Consider special offers or engagement campaigns."
                )

            elif cluster == 1:
                st.info(
                    "These are regular customers with consistent purchasing behavior."
                )

            elif cluster == 0:
                st.info(
                    "These customers shop occasionally and may need promotional offers to increase engagement."
                )

        else:

            st.info("Enter the customer details and click **Find Customer Segment**.")

    st.write("")

    with st.expander("Model Information"):

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Algorithm**")
            st.write("K-Means Clustering")

            st.write("**Features Used**")
            st.write("""
- Recency
- Frequency
- Monetary (RFM)
""")

        with col2:

            st.write("**Dataset**")
            st.write("Online Retail Dataset")

            st.write("**Purpose**")
            st.write(
                "Group customers based on their purchasing behaviour for targeted marketing."
            )


# =====================================================
# CAT VS DOG CLASSIFICATION
# =====================================================

elif project == "Cat vs Dog Classification":

    st.header("Cat vs Dog Classification")
    st.write("Upload an image of a cat or dog to classify it using the trained SVM model.")

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="catdog"
        )

        predict = False

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            predict = st.button("Predict Animal")

    with right:

        st.subheader("Prediction")

        if uploaded_file is not None and predict:

            # Preprocess image
            image = image.convert("L")
            image = image.resize((64, 64))

            img = np.array(image)

            img = img.flatten().reshape(1, -1)

            # Scale image
            img = svm_scaler.transform(img)

            # Predict
            prediction = svm_model.predict(img)[0]

            if prediction == 0:
                result = "Cat"
            else:
                result = "Dog"

            st.markdown(f"""
            <div class="result-card">
                <h3>Prediction Result</h3>
                <h2>{result}</h2>
            </div>
            """, unsafe_allow_html=True)

        elif uploaded_file is None:

            st.info("Upload an image to begin prediction.")

        else:

            st.info("Click **Predict Animal** to classify the uploaded image.")

    st.write("")

    with st.expander("Model Information"):

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Algorithm**")
            st.write("Support Vector Machine (SVM)")

            st.write("**Image Size**")
            st.write("64 × 64 Grayscale")

        with col2:

            st.write("**Dataset**")
            st.write("Dog and Cat Classification Dataset")

            st.write("**Purpose**")
            st.write("Classify an uploaded image as either a cat or a dog.")

# =====================================================
# HAND GESTURE RECOGNITION
# =====================================================

elif project == "Hand Gesture Recognition":

    st.header("Hand Gesture Recognition")
    st.write("Upload a hand gesture image to predict the gesture using the trained CNN model.")

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Upload Image")

        uploaded_file = st.file_uploader(
            "Choose a hand gesture image",
            type=["jpg", "jpeg", "png"],
            key="gesture"
        )

        predict = False

        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            predict = st.button("Predict Gesture")

    with right:

        st.subheader("Prediction")

        if uploaded_file is not None and predict:

            # Preprocess image
            img = image.resize((64, 64))
            img = np.array(img, dtype=np.float32)

            img = img / 255.0

            img = np.expand_dims(img, axis=0)

            # Predict
            prediction = gesture_model.predict(img, verbose=0)

            predicted_class = np.argmax(prediction)

            gesture_names = {
                0: "Palm",
                1: "L",
                2: "Fist",
                3: "Fist Moved",
                4: "Thumb",
                5: "Index",
                6: "OK",
                7: "Palm Moved",
                8: "C",
                9: "Down"
            }

            confidence = float(np.max(prediction))

            st.markdown(f"""
            <div class="result-card">
                <h3>Predicted Gesture</h3>
                <h2>{gesture_names[predicted_class]}</h2>
                <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence)

        elif uploaded_file is None:

            st.info("Upload a hand gesture image to begin prediction.")

        else:

            st.info("Click **Predict Gesture** to classify the uploaded image.")

    st.write("")

    with st.expander("Model Information"):

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Algorithm**")
            st.write("Convolutional Neural Network (CNN)")

            st.write("**Image Size**")
            st.write("64 × 64 RGB")

        with col2:

            st.write("**Dataset**")
            st.write("LeapGestRecog")

            st.write("**Purpose**")
            st.write("Recognize hand gestures from uploaded images.")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:10px;">
        Machine Learning Internship Dashboard<br>
        Built using Streamlit, Scikit-learn and TensorFlow
    </div>
    """,
    unsafe_allow_html=True
)