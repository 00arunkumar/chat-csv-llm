#Import required libraries


import streamlit as st
import pandas as pd
from openai import RateLimitError


from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent






#Title
st.title('CSVInsights📊 🤖')

#Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data exploration.")

#Explanation sidebar
with st.sidebar:
    api_key = st.text_input('Please enter your OpenAI API key')

    st.write('*Your Data Science Adventure Begins with an CSV File.*')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> made with ❤️ </p>",unsafe_allow_html=True )

#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        #llm model
        llm = OpenAI(openai_api_key=api_key, temperature = 0)

       

        #Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)

        #Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            #correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            #st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y =[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return
        


        #Main
        try:
            st.header('Exploratory data analysis')
            st.subheader('General information about the dataset')
            function_agent()
            
            st.subheader('Variable of study')
            user_question_variable = st.selectbox("what variables are you interested in ?", df.columns)
            if user_question_variable is not None and user_question_variable !="":
                function_question_variable()
                
            st.subheader('Further study')
            
            st.subheader('Suggestion Queries for your CSV file')
            sugg_query = pandas_agent.run("Give me 5 best Analytical question which i can ask from this upload csv data ?.")
            st.write(sugg_query)
            
            if user_question_variable:
                user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
                if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                    function_question_dataframe()
                if user_question_dataframe in ("no", "No"):
                    st.write("")
        except RateLimitError :
            st.error("You had used all your free tokens of your OpenAI API. Please enter a different OpenAI API Key.", icon="🚨")
            st.image("https://static-00.iconduck.com/assets.00/robot-404-error-errors-illustration-2048x1704-clz3f1n7.png")



