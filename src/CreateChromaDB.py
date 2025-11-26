from time import sleep
from pathlib import Path
import json

from typing import List
from httpx import ReadError
from alive_progress import alive_it
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_mistralai.embeddings import MistralAIEmbeddings



class CreateChromaDB:

    def __init__(
        self, 
        embeddings,
        chunked_docs_dir: Path | str, 
        chroma_db_dir: Path | str, 
        checkpoint_file: Path | str = "embedding_progress.json",
        failed_batches_file: Path | str = "failed_batches.txt",
        collection_name: str = "default_collection",
        ):
       
        """
        Create a ChromaDB vector database from chunked JSONL files in a specified directory.

        Args:
            embeddings: Embedding function to use for vectorization.
            chunked_docs_dir (Path | str): Directory containing markdown files.
            chroma_db_dir (Path | str): Directory to store the ChromaDB database.
            checkpoint_file (str): File to store processing progress.
            failed_batches_file (str): File to log failed batches.
            collection_name (str): Name of the ChromaDB collection.
        """

        self.embeddings = embeddings
        self.collection_name = collection_name
        self.chunked_docs_dir = Path(chunked_docs_dir)
        self.chroma_db_dir = Path(chroma_db_dir)
        self.checkpoint_file = Path(checkpoint_file)
        self.failed_batches_file = Path(failed_batches_file)

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

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_db_dir)
        )


    def load_chunked_documents(self):
        """
        Recursively load chunked JSONL documents from the specified directory.

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
    

    def load_checkpoint(self):
        """
        Load the last checkpoint to resume processing.
        
        Returns:
            int: The index to start processing from.
        """
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_idx = checkpoint['last_processed']
                print(f"Resuming from index {start_idx}")
                return start_idx
        except FileNotFoundError:
            print("No checkpoint found. Starting from beginning.")
            return 0


    def save_checkpoint(self, index: int, total_docs: int, batch_size: int):
        """
        Save the current processing checkpoint.
        
        Args:
            index (int): Current processing index.
            total_docs (int): Total number of documents.
            batch_size (int): Batch size being used.
        """
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'last_processed': index,
                'total_docs': total_docs,
                'batch_size': batch_size
            }, f)


    def log_failed_batch(self, batch_num: int, start_idx: int, end_idx: int, error: Exception):
        """
        Log details of a failed batch.
        
        Args:
            batch_num (int): Batch number.
            start_idx (int): Starting index of the batch.
            end_idx (int): Ending index of the batch.
            error (Exception): The error that occurred.
        """
        with open(self.failed_batches_file, 'a') as f:
            f.write(f"Batch {batch_num}: indices {start_idx}-{end_idx}\n")
            f.write(f"Error: {str(error)}\n")
            f.write("-" * 50 + "\n")


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ReadError,))
    )
    def add_batch_with_retry(self, batch: List[Document]):
        """
        Add a batch of documents with retry logic.
        
        Args:
            batch: List of documents to add.
        """
        self.vectorstore.add_documents(batch)


    def embed_and_store(self, 
                        batch_size: int = 100, 
                        delay_seconds: float = 1,
                        resume: bool = True):
        """
        Embed and store the loaded documents into the ChromaDB vector database in batches.
        
        Args:
            batch_size (int): Number of documents to process in each batch. Default is 100.
            delay_seconds (float): Seconds to wait between batches. Default is 1.
            resume (bool): Whether to resume from checkpoint. Default is True.
        """
        
        # Load documents
        documents = self.load_chunked_documents()
        print(f"Loaded {len(documents)} documents")
        
        # Determine starting index
        start_idx = self.load_checkpoint() if resume else 0
        
        # Calculate total batches for progress tracking
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        try:
            for i in range(start_idx, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                print(f"Processing batch {batch_num}/{total_batches} (indices {i}-{i+len(batch)-1})")
                
                try:
                    self.add_batch_with_retry(batch)
                    
                    # Save checkpoint after successful batch
                    self.save_checkpoint(i + batch_size, len(documents), batch_size)
                    
                    # Rate limiting delay
                    sleep(delay_seconds)
                    
                except Exception as e:
                    print(f"\nBatch {batch_num} failed after retries: {e}")
                    
                    # Log failed batch
                    self.log_failed_batch(batch_num, i, i + len(batch) - 1, e)
                    
                    print(f"Logged to {self.failed_batches_file}. Continuing with next batch...")
                    
                    # Update checkpoint to skip this batch
                    self.save_checkpoint(i + batch_size, len(documents), batch_size)
                    
                    # Longer delay after failure
                    sleep(5)
                    continue

        except KeyboardInterrupt:
            print(f"\n\nProcess interrupted. Progress saved at index {i}.")
            print(f"Run again with resume=True to continue from this point.")
            raise

        print("\n✓ Embedding complete!")
        print(f"Processed {len(documents)} documents")
        
        # Check for failures
        try:
            with open(self.failed_batches_file, 'r') as f:
                failures = f.read()
                if failures:
                    print(f"\n⚠ Some batches failed. Check {self.failed_batches_file} for details.")
        except FileNotFoundError:
            print("No failures recorded.")


    def embed_and_store_legacy(self, 
                        batch_size: int = 100, 
                        refresh_interval: int = 5, 
                        max_exceptions: int = 5,
                        start_index: int = 0):
        """
        Legacy method: Embed and store using recursive approach.
        
        Args:
            batch_size (int): Number of documents to process in each batch. Default is 100.
            refresh_interval (int): Number of batches before refreshing embeddings. Default is 5.
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
    creator = CreateChromaDB(
        embeddings=embeddings,
        collection_name="perma_rag_collection",
        chunked_docs_dir=Path("../data/chunked_documents/"),
        chroma_db_dir=Path("../chroma_db"),
        checkpoint_file=Path("../logs/embedding_progress.json"),
        failed_batches_file=Path("../logs/failed_batches.txt")
    )

    # Use the new method with checkpointing and retry logic
    creator.embed_and_store(batch_size=100, delay_seconds=1, resume=True)
    
    # Or use the legacy method if preferred
    # creator.embed_and_store_legacy(start_index=0)
    
    print("ChromaDB creation complete.")
