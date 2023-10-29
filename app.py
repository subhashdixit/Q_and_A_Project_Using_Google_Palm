# Import necessary libraries
import streamlit as st
from src.langchain_helper import get_qa_chain, create_vector_db

# Set the title of the Streamlit app
st.title("Codebasics Q&A 🌱")

# Create a button to trigger the creation of a knowledgebase
btn = st.button("Create Knowledgebase")

# Check if the "Create Knowledgebase" button is pressed
if btn:
    # Call the function to create the vector database (knowledgebase)
    create_vector_db()

# Create a text input field for users to enter their questions
question = st.text_input("Question: ")

# Check if a question has been entered by the user
if question:
    # Call the function to retrieve a question-answering chain
    chain = get_qa_chain()

    # Use the chain to find an answer to the user's question
    response = chain(question)

    # Display a header for the answer
    st.header("Answer")

    # Display the answer obtained from the question-answering chain
    st.write(response["result"])
