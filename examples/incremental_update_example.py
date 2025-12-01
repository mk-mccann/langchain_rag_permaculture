"""
Example demonstrating incremental update functionality of CreateChromaDB class.

This script shows how to:
1. Initial database creation
2. Add new documents and update automatically
3. Modify existing documents and update only changed portions
4. Force full rebuild when needed
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai.embeddings import MistralAIEmbeddings
import sys

# Add parent directory to path to import CreateChromaDB
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.CreateChromaDB import CreateChromaDB


def main():
    # Load environment variables
    load_dotenv()
    api_key_str = os.getenv("MISTRAL_API_KEY")
    
    if not api_key_str:
        raise ValueError("MISTRAL_API_KEY not found in environment variables")
    
    api_key_str = api_key_str.strip()

    # Setup Mistral embeddings
    embeddings = MistralAIEmbeddings(api_key=api_key_str, model="mistral-embed")

    # Initialize CreateChromaDB with incremental update support
    creator = CreateChromaDB(
        embeddings=embeddings,
        collection_name="perma_rag_collection",
        chunked_docs_dir=Path("../data/chunked_documents/"),
        chroma_db_dir=Path("../chroma_db"),
        checkpoint_file=Path("../logs/embedding_progress.json"),
        failed_batches_file=Path("../logs/failed_batches.txt"),
        index_file=Path("../data/chunked_documents/index.json")  # Optional, defaults to chunked_docs_dir/index.json
    )

    print("=" * 80)
    print("INCREMENTAL UPDATE EXAMPLES")
    print("=" * 80)

    # Example 1: Incremental update (default behavior)
    print("\n1. Running incremental update (processes only new/modified files)...")
    print("-" * 80)
    creator.embed_and_store(
        batch_size=100,
        delay_seconds=1,
        resume=True,
        incremental=True  # This is the default
    )

    # Example 2: Full rebuild from scratch
    print("\n2. To force a full rebuild, set incremental=False...")
    print("-" * 80)
    # Uncomment to run full rebuild:
    # creator.embed_and_store(
    #     batch_size=100,
    #     delay_seconds=1,
    #     resume=False,
    #     incremental=False
    # )
    print("(Skipped - uncomment in code to run)")

    # Example 3: Load only specific files
    print("\n3. You can also load specific files programmatically...")
    print("-" * 80)
    # Get new and modified files
    new_files, modified_files = creator.get_modified_and_new_files()
    print(f"New files: {len(new_files)}")
    print(f"Modified files: {len(modified_files)}")
    
    if new_files:
        print("\nNew files found:")
        for f in list(new_files)[:5]:  # Show first 5
            print(f"  - {f}")
        if len(new_files) > 5:
            print(f"  ... and {len(new_files) - 5} more")

    print("\n" + "=" * 80)
    print("WORKFLOW RECOMMENDATIONS")
    print("=" * 80)
    print("""
    For regular updates (recommended):
    - Run with incremental=True (default)
    - The system will automatically:
      * Detect new files from your chunking process
      * Detect modified files by comparing with index.json
      * Remove old embeddings for modified files
      * Add embeddings only for new/modified content
    
    For a fresh start:
    - Delete the ChromaDB directory
    - Run with incremental=False
    - This will rebuild the entire database from scratch
    
    For troubleshooting:
    - Check logs/embedding_progress.json for progress
    - Check logs/failed_batches.txt for any errors
    - The system will resume from the last checkpoint if interrupted
    """)

    print("\nDatabase update complete!")


if __name__ == "__main__":
    main()
