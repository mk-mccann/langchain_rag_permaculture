from pathlib import Path

from alive_progress import alive_it
from mistralai import Mistral
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_mistralai.embeddings import MistralAIEmbeddings


class createChromaDB:

    def __init__(
        self, 
        embeddings,
        chunked_docs_dir: Path | str, 
        chroma_db_dir: Path | str, 
        collection_name: str = "default_collection"
        ):
       
        """
        Create a ChromaDB vector database from chunked JSONL files in a specified directory.

        Args:
            embeddings: Embedding function to use for vectorization.
            chunked_docs_dir (Path | str): Directory containing markdown files.
            chroma_db_dir (Path | str): Directory to store the ChromaDB database.
            collection_name (str): Name of the ChromaDB collection.
        """

        self.chunked_docs_dir = Path(chunked_docs_dir)
        self.chroma_db_dir = Path(chroma_db_dir)

        if not self.chroma_db_dir.exists():
            self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma(
            collection_name = collection_name,
            embedding_function = embeddings,
            persist_directory = str(self.chroma_db_dir)  # Where to save data locally, remove if not necessary
        )    

        self.client = Chroma(persist_directory=str(self.chroma_db_dir))


    def load_chunked_documents(self):
        """
        Load chunked JSONL documents from the specified directory.

        Returns:
            List of loaded documents.
        """

        all_docs = []
        jsonl_files = list(self.chunked_docs_dir.glob("**/*.jsonl"))

        for file_path in alive_it(jsonl_files, title="Loading chunked documents for ChromaDB"):
            loader = JSONLoader(
                file_path=str(file_path),     
                jq_schema='.',
                content_key='page_content',
                json_lines=True,
                metadata_func=lambda record, metadata: record.get('metadata', {}))
                
            docs = loader.load()
            docs = filter_complex_metadata(docs)
            all_docs.extend(docs)

        # Remove any leading/trailing whitespace from document contents and empty entries
        all_docs = [doc for doc in all_docs if doc.page_content.strip()]

        return all_docs
    

    def embed_and_store(self):
        """
        Embed and store the loaded documents into the ChromaDB vector database.
        """

        documents = self.load_chunked_documents()

        print(f"Embedding and storing {len(documents)} documents into ChromaDB...")
        self.vectorstore.add_documents(documents)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY").strip()

    # Setup Mistral model and embeddings
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key, model="mistral-embed")

    # Setup and create ChromaDB
    creator = createChromaDB(
        embeddings=embeddings,
        chunked_docs_dir=Path("../data/chunked_documents/"),
        chroma_db_dir=Path("../chroma_db"),
        collection_name="perma_rag_collection"
    )

    creator.embed_and_store()
    print("ChromaDB creation complete.")
