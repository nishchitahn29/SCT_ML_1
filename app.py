import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------- Sample Dataset ----------------
data = pd.DataFrame({
    'square_feet': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 1200, 2000, 1750],
    'bedrooms': [3, 3, 4, 4, 2, 3, 5, 2, 4, 3],
    'bathrooms': [2, 2, 3, 2, 1, 2, 3, 1, 2, 2],
    'price': [250000, 290000, 320000, 340000, 200000, 275000, 450000, 210000, 380000, 310000]
})

# ---------------- Train Model ----------------
X = data[['square_feet', 'bedrooms', 'bathrooms']]
y = data['price']
model = LinearRegression()
model.fit(X, y)

# ---------------- Streamlit UI ----------------
st.title("🏠 House Price Prediction App")
st.write("Predict house prices based on square footage, bedrooms, and bathrooms.")

square_feet = st.number_input("Square Feet", min_value=500, max_value=5000, value=1500)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)

if st.button("Predict Price"):
    predicted_price = model.predict([[square_feet, bedrooms, bathrooms]])
    st.success(f"💰 Predicted House Price: ${predicted_price[0]:,.2f}")
