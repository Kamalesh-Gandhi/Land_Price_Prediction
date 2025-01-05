import streamlit as st
import joblib
import pandas as pd

# Set the page configuration
st.set_page_config(
    page_title="Chennai Land Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load the trained model
model = joblib.load('xgboost_best_model.pkl')

# Load the encoders
try:
    encoders = joblib.load('encode_categorical_columns.pkl')
except FileNotFoundError:
    st.error("The encoding filei 'encode_categorical_columns.pkl' is missing.")
    st.stop()

# Verify keys in the encoders dictionary
required_keys = ['AREA', 'SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'MZZONE']
for key in required_keys:
    if key not in encoders:
        st.error(f"Missing encoder for '{key}' in 'encode_categorical_columns.pkl'.")
        st.stop()

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the CSS
local_css("style.css")


# Add a sidebar for navigation
st.sidebar.title("Chennai Land Price Checker")
st.sidebar.image(
    "Chennai_Price_Prediction_logo.webp", caption="Know Your Property's Worth", use_container_width=True
)

# Main page title
st.markdown(
    """
    <h1 style="text-align: center; color: #2B579A;">Chennai Land Price Prediction 🏠</h1>
    <p style="text-align: center; font-size: 18px; color: #555555;">Enter the property details below to estimate its value.</p>
    """,
    unsafe_allow_html=True
)

# Create a form layout for user inputs
with st.form("land_price_form"):
    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        AREA = st.selectbox("Area", encoders['AREA'].classes_)
        INT_SQFT = st.number_input("Interior Square Footage (e.g., 1500)", min_value=100, max_value=10000, step=10)
        DIST_MAINROAD = st.number_input("Distance from Main Road (meters)", min_value=0, max_value=1000, step=1)
        N_BEDROOM = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
        N_BATHROOM = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
        N_ROOM = st.number_input("Total Number of Rooms", min_value=1, max_value=20, step=1)
        REG_FEE = st.number_input("Registration Fee (INR)", min_value=0, step=1000)
        COMMIS = st.number_input("Commission (INR)", min_value=0, step=1000)
        Age_of_building = st.number_input("Age of the Building (Years)", min_value=0, max_value=100, step=1)

    with col2:
        SALE_COND = st.selectbox("Sale Condition", encoders['SALE_COND'].classes_)
        PARK_FACIL = st.selectbox("Parking Facility", encoders['PARK_FACIL'].classes_)
        BUILDTYPE = st.selectbox("Building Type", encoders['BUILDTYPE'].classes_)
        UTILITY_AVAIL = st.selectbox("Utility Availability", encoders['UTILITY_AVAIL'].classes_)
        STREET = st.selectbox("Street Type", encoders['STREET'].classes_)
        MZZONE = st.selectbox("Zone Type", encoders['MZZONE'].classes_)
        QS_ROOMS = st.number_input("Room Quality Score (e.g., 4.5)", format="%.2f", min_value=0.0, max_value=5.0, step=0.1)
        QS_BATHROOM = st.number_input("Bathroom Quality Score (e.g., 4.0)", format="%.2f", min_value=0.0, max_value=5.0, step=0.1)
        QS_BEDROOM = st.number_input("Bedroom Quality Score (e.g., 4.2)", format="%.2f", min_value=0.0, max_value=5.0, step=0.1)
        QS_OVERALL = st.number_input("Overall Quality Score (e.g., 4.3)", format="%.2f", min_value=0.0, max_value=5.0, step=0.1)
        

    # Submit button
    submitted = st.form_submit_button("Predict")

# When the form is submitted
if submitted:
    # Encode categorical inputs using the encoders
    encoded_inputs = {
        'AREA': encoders['AREA'].transform([AREA])[0],
        'SALE_COND': encoders['SALE_COND'].transform([SALE_COND])[0],
        'PARK_FACIL': encoders['PARK_FACIL'].transform([PARK_FACIL])[0],
        'BUILDTYPE': encoders['BUILDTYPE'].transform([BUILDTYPE])[0],
        'UTILITY_AVAIL': encoders['UTILITY_AVAIL'].transform([UTILITY_AVAIL])[0],
        'STREET': encoders['STREET'].transform([STREET])[0],
        'MZZONE': encoders['MZZONE'].transform([MZZONE])[0],
    }

    # Combine all inputs into a single list
    input_data = [
        [
            encoded_inputs['AREA'], INT_SQFT, DIST_MAINROAD, N_BEDROOM, N_BATHROOM, N_ROOM,
            encoded_inputs['SALE_COND'], encoded_inputs['PARK_FACIL'], encoded_inputs['BUILDTYPE'],
            encoded_inputs['UTILITY_AVAIL'], encoded_inputs['STREET'], encoded_inputs['MZZONE'],
            QS_ROOMS, QS_BATHROOM, QS_BEDROOM, QS_OVERALL, REG_FEE, COMMIS, Age_of_building
        ]
    ]

    # Prepare input as DataFrame for the model
    input_df = pd.DataFrame(input_data, columns=[
        'AREA', 'INT_SQFT', 'DIST_MAINROAD', 'N_BEDROOM', 'N_BATHROOM',
        'N_ROOM', 'SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'UTILITY_AVAIL',
        'STREET', 'MZZONE', 'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM',
        'QS_OVERALL', 'REG_FEE', 'COMMIS', 'Age_of_building'
    ])

    # Make the prediction
    try:
        prediction = model.predict(input_df)

        # Display results
        st.markdown(
            f"""
            <div style="background-color: #DFF6DD; padding: 10px; border-radius: 5px; border: 1px solid #4CAF50;">
                <h3 style="text-align: center; color: #388E3C;">Predicted Price</h3>
                <p style="text-align: center; font-size: 24px; color: #2E7D32;">₹{prediction[0]:,.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"An error occurred: {e}") 

# Footer
st.sidebar.markdown(
    """
    ---
    **Powered by KANI Real ESTATE**  
    Contact us: [kaniRealEst@gmail.com](mailto:kaniRealEst@gmail.com)
    """
)
