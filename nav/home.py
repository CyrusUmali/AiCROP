import streamlit as st

def show():
    st.title("Crop Recommendation System")
    st.write("""
        Welcome to the Crop Recommendation System. This tool helps farmers determine the best crop to plant based on various parameters.
        Use the navigation bar to explore different sections:
        - **Home**: Overview of the system.
        - **Predict**: Enter parameters to get crop recommendations.
        - **Visualize**: View model metrics and decision tree visualization.
    """)
    
    # Displaying the image
    st.image("images/home.webp", use_column_width=True, caption="Home Image")
