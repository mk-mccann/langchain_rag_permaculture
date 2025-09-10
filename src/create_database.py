from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.schema import Document
import mistralai
import dotenv
import uuid
import os


# Set the system MistralAI API key
mistralai.api_key = os.environ["MISTRAL_API_KEY"]

# Set paths to data source and local chroma database
DATA_PATH = "./data/books"
CHROMA_PATH = "./chroma"


def load_documents(path: str) -> list[Document]:
    '''
    This function loads all markdown files from the specified directory and returns them as a list of Document objects.
    '''

    loader = DirectoryLoader(path, glob="*.md")
    documents = loader.load()
    return documents


def load_pdf_documents(path: str) -> list[Document]:
    '''
    This function loads all PDF files from the specified directory and returns them as a list of Document objects.
    '''

    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def split_documents(documents:list[Document], splitter='md') -> list[Document]:
    '''
    Here I want to split the document(s) into smaller chunks of semanitically related text.
    I'm choosing to use the MarkdownHeaderTextSplitter by default to split the documents by headers, since I'm using a 
    .md format for my books. However, if the documents were in a different format, I could use 
    the RecursiveCharacterTextSplitter to split based on characters, sentences, or paragraphs.  
    '''

    if splitter == 'md':
        text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                "#", "##", "###", "####", "#####", "######"
            ],
            chunk_size=500,
            chunk_overlap=50
        )
    else:
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
