from time import sleep
from pathlib import Path

from httpx import ReadError
from alive_progress import alive_it
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
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
        self.collection_name = collection_name

        if not self.chroma_db_dir.exists():
            self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma(
            collection_name = self.collection_name,
            embedding_function = embeddings,
            persist_directory = str(self.chroma_db_dir)
        )    


    def refresh_embeddings(self):
        """
        Refresh embeddings in the ChromaDB vector database.
        """

        embeddings = MistralAIEmbeddings(model="mistral-embed")
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.chroma_db_dir)
        )


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
    

    def embed_and_store(self, 
                        batch_size: int = 500, 
                        refresh_interval: int = 4, 
                        max_exceptions: int = 5,
                        start_index: int = 0):
        """
        Embed and store the loaded documents into the ChromaDB vector database in batches.
        
        Args:
            batch_size (int): Number of documents to process in each batch. Default is 500.
            refresh_interval (int): Number of batches before refreshing embeddings. Default is 4.
            max_exceptions (int): Maximum number of exceptions before quitting. Default is 5.
            start_index (int): Index to start processing from. Default is 0.
        """

        def recursive_embed(documents, batch_size, refresh_interval, start_index=0, exception_count=0):
            if exception_count >= max_exceptions:
                print(f"Maximum exceptions ({max_exceptions}) reached. Quitting.")
                print(f"Last processed batch: {start_index//batch_size}")
                return
            
            refresh_counter = start_index // batch_size % refresh_interval
            
            try:
                for i in range(start_index, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
                    print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                    sleep(5)

                    refresh_counter += 1
                    if refresh_counter >= refresh_interval:
                        refresh_counter = 0
                        print("Refreshing embeddings to manage resources...")
                        self.refresh_embeddings()
                        
            except ReadError as e:
                exception_count += 1
                print(f"ReadError occurred at batch {i//batch_size + 1}")
                print(f"Exception count: {exception_count}/{max_exceptions}")
                print("Cooling down for 5 minutes before retrying...")
                sleep(5*60)
                self.refresh_embeddings()
                print(f"Resuming from batch {i//batch_size + 1}...")
                recursive_embed(documents, batch_size, refresh_interval, start_index=i, exception_count=exception_count)

        documents = self.load_chunked_documents()
        print(f"Embedding and storing {len(documents)} documents into ChromaDB...")
        recursive_embed(documents, batch_size, refresh_interval, start_index=start_index)


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

    creator.embed_and_store(start_index=54)
    print("ChromaDB creation complete.")
