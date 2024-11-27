import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

def setup_qa_system():
    """
    Initialize the QA system with existing Pinecone index.
    Returns the QA chain ready for querying.
    """
    # Get API keys from Streamlit secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    index_name = st.secrets["PINECONE_INDEX_NAME"]
    
    # Set up environment variables (if needed)
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    
    # Initialize components
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Connect to existing vector store
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return qa_chain, vectorstore

def search_documents(vectorstore, query, k=4):
    """
    Search for relevant documents using similarity search.
    
    Args:
        vectorstore: Initialized vector store
        query (str): Search query
        k (int): Number of documents to retrieve
        
    Returns:
        List of relevant documents
    """
    return vectorstore.similarity_search(query, k=k)

def ask_question(qa_chain, question, vectorstore):
    """
    Ask a question and get an answer using the QA chain after fetching relevant documents.
    
    Args:
        qa_chain: Initialized QA chain
        question (str): Question to ask
        vectorstore: Initialized vector store
        
    Returns:
        str: Answer to the question
    """
    # Fetch relevant documents
    relevant_docs = search_documents(vectorstore, question, k=5)
    
    # Combine the documents' content to form a context for the LLM
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Pass the context along with the question to the QA chain
    response = qa_chain({"query": f"{context}\n\n{question}"})
    return response["result"]

def main():
    # Set up the QA system
    qa_chain, vectorstore = setup_qa_system()
    
    st.title("QA System")
    
    option = st.selectbox("Choose an option:", ["Search documents", "Ask a question", "Exit"])
    
    if option == "Search documents":
        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            results = search_documents(vectorstore, query)
            st.subheader("Search Results:")
            for i, doc in enumerate(results, 1):
                st.write(f"**Document {i}:**")
                st.write(f"{doc.page_content[:200]}...")

    elif option == "Ask a question":
        question = st.text_input("Enter your question:")
        if st.button("Ask"):
            answer = ask_question(qa_chain, question, vectorstore)
            st.subheader("Answer:")
            st.write(answer)

    elif option == "Exit":
        st.write("Goodbye!")

if __name__ == "__main__":
    main()
