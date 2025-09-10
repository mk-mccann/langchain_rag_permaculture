from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.schema import Document
from mistralai import Mistral
import dotenv
import uuid
import os
import asyncio


# Load the environment variables and set the system MistralAI API key
dotenv.load_dotenv()
mistral_api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=mistral_api_key)

# Set paths to data source(s) and local chroma database
DATA_PATH = "./data/books"
CHROMA_PATH = "./chroma"


def load_documents(path: str) -> list[Document]:
    '''
    This function loads all files from the specified directory and returns them as a list of Document objects.
    Here I'm using the DirectoryLoader to load all files from the specified directory without needing to 
    specify the filetype. By default I show a progress bar and use multithreading to speed up the loading process.
    '''
    loader = DirectoryLoader(path, show_progress=True, use_multithreading=True)
    documents = loader.load()
    return documents


def split_documents(documents:list[Document]) -> list[Document]:
    '''
    Here I want to split the document(s) into smaller chunks of semantically related text.
    I'm choosing to use the MarkdownHeaderTextSplitter by default to split the documents by headers, 
    since I'm using a .md format for my books. However, if the documents were in a different format, 
    I could use the RecursiveCharacterTextSplitter to split based on characters, sentences, or paragraphs.
    '''

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
        )


def create_chroma_database(documents: list[Document], persist_directory: str):

    # Create embeddings
    embeddings = MistralAIEmbeddings()

    # Create a unique collection name using UUID
    collection_name = f"books_collection_{uuid.uuid4()}"

    # Create Chroma vector store
    vectordb = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    # Persist the database to disk
    vectordb.persist()
    return vectordb
