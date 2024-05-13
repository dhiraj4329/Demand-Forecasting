import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
from pycaret.regression import load_model, predict_model
import plotly.graph_objects as go
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Load the trained model
model = load_model('final')

# Load the DataFrame
df = pd.read_csv('C:/Users/DhirajVelhal/Documents/Projects/supply_forecasting/Data/df_final.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Convert DataFrame to SQLite database
conn = sqlite3.connect('C:/Users/DhirajVelhal/Documents/Projects/supply_forecasting/Data/df_final.db')
df.to_sql('df_final', conn, if_exists='replace', index=False)
cur = conn.cursor()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define prompt for Gemini model
prompt = [
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name df_final and has the following columns -'Unnamed: 0', 'GRADE', 'temp', 
    'dew', 'humidity', 'precip', 'Year','Month', 'Week', 'Holiday', 'Actual Quantity', 'Predicted Quantity'
    \n\nFor example,\nExample 1 -show me the actual and predicted quantity of grade 
    Fancy Domestic from week 24 to 28 for year 2023, 
    the SQL command will be something like this SELECT "Actual Quantity", "Predicted Quantity" 
    FROM df_final WHERE GRADE = 'Fancy Domestic' AND Year = 2023 AND Week BETWEEN 24 AND 28; ;
    \n\nExample 2 - What is the average temperature?,
    the SQL command will be something like this SELECT AVG(temp) FROM df_final ;
    also the sql code should not have ``` in beginning or end and sql word in output
    show the output in the form of a table with the header as related columns

    """
]

# Function to get response from Gemini model
def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

# Function to predict quantity
def predict_quantity(row, grade, year, week):
    feature_names = ['temp', 'humidity', 'precip', 'dew', 'GRADE', 'Year', 'Month', 'Week', 'Holiday']
    input_data = pd.DataFrame({
        'temp': [row['temp']],
        'humidity': [row['humidity']],
        'precip': [row['precip']],
        'dew': [row['dew']],
        'GRADE': [grade],
        'Year': [year],
        'Month': [row['Month']],
        'Week': [week],
        'Holiday': [row['Holiday']]
    })
    input_data = input_data[feature_names]
    prediction = model.predict(input_data)[0]
    return prediction

# Main function for Streamlit app
def main():
    # Display the image
    image = Image.open('C:/Users/DhirajVelhal/Documents/Projects/supply_forecasting/wonderful.jpg')
    st.image(image)

    # Set the title of the Streamlit app
    #st.markdown("<h1 style='text-align: left; color: Black;'>Quantity Prediction  App</h1>", unsafe_allow_html=True)

    # User input for the question
    question = st.text_input('Enter the question')

    if st.button('Get Response'):
        response = get_gemini_response(question, prompt)

        data = cur.execute(response)
        df_output = pd.DataFrame(data.fetchall(), columns=[i[0] for i in cur.description])
        st.write(df_output)

    # Sidebar title
    st.sidebar.markdown("<h2 style='text-align: center; color: green;'>Quantity Prediction</h2>", unsafe_allow_html=True)

    # Sidebar inputs for the quantity prediction
    start_week = st.sidebar.selectbox('Start Week', sorted(df['Week'].unique()))
    end_week = st.sidebar.selectbox('End Week', sorted(df['Week'].unique()))
    start_year = st.sidebar.selectbox('Start Year', sorted(df['Year'].unique()))
    end_year = st.sidebar.selectbox('End Year', sorted(df['Year'].unique()))
    grade = st.sidebar.selectbox('GRADE', sorted(df['GRADE'].unique()))

    data = df[(df['Week'] >= start_week) & (df['Week'] <= end_week) & 
              (df['Year'] >= start_year) & (df['Year'] <= end_year) & 
              (df['GRADE'] == grade)]

    output_data = pd.DataFrame(columns=['Week', 'Year', 'GRADE', 'Actual Quantity', 'Predicted Quantity'])

    if st.sidebar.button('Predict'):
        for index, row in data.iterrows():
            year = row['Year']
            week = row['Week']
            month = row['Month']
            prediction = predict_quantity(row, grade, year, week)

            actual_value = None
            filtered_row_train = df[(df['Year'] == year) & (df['Week'] == week) & (df['GRADE'] == grade)]
            if not filtered_row_train.empty:
                actual_value = filtered_row_train.iloc[0]['Actual Quantity']  # Fetch actual quantity here

            output_data = output_data.append({
                'Week': week,
                'Month':month,
                'Year': str(year),
                'GRADE': grade,
                'Actual Quantity': (actual_value) if actual_value is not None else None,
                'Predicted Quantity': int(prediction) if prediction is not None else 0
            }, ignore_index=True)

        st.dataframe(output_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=output_data.index, y=output_data['Actual Quantity'], mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=output_data.index, y=output_data['Predicted Quantity'], mode='lines', name='Predicted', line=dict(color='red')))
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
