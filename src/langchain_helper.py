# Import necessary libraries and modules
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv

# Load environment variables from a .env file (particularly an OpenAI API key)
load_dotenv()

# Create a Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Specify the file path for the FAISS vector database
vectordb_file_path = "faiss_index"

# Function to create and save a vector database
def create_vector_db():
    # Load data from a CSV file (presumably containing FAQs)
    path = "Data\codebasics_faqs.csv"
    
    loader = CSVLoader(file_path=path, source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for a vector database from the 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save the vector database locally
    vectordb.save_local(vectordb_file_path)

# Function to set up a question-answering chain
def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database with a score threshold
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Define a prompt template for generating answers based on context and questions
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    # Create a PromptTemplate with input variables
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create a RetrievalQA chain with specified parameters
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

# Main script entry point
if __name__ == "__main__":
    # Create the vector database
    create_vector_db()

    # Get the question-answering chain
    chain = get_qa_chain()

    # Query the chain with a specific question and print the result
    print(chain("Do you have a JavaScript course?"))
