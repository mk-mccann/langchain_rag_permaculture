import pytest
from pathlib import Path
import json
import tempfile
from src.createChromaDB import createChromaDB

@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as chunked_dir, \
         tempfile.TemporaryDirectory() as chroma_dir:
        yield Path(chunked_dir), Path(chroma_dir)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings function."""
    class MockEmbeddings:
        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]
        
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]
    
    return MockEmbeddings()


@pytest.fixture
def sample_jsonl_file(temp_dirs):
    """Create a sample JSONL file for testing."""
    chunked_dir, _ = temp_dirs
    jsonl_path = chunked_dir / "test_chunks.jsonl"
    
    test_data = [
        {
            "page_content": "This is the first chunk of text.",
            "metadata": {"source": "doc1.md", "chunk_id": 1}
        },
        {
            "page_content": "This is the second chunk of text.",
            "metadata": {"source": "doc1.md", "chunk_id": 2}
        },
        {
            "page_content": "This is the third chunk of text.",
            "metadata": {"source": "doc2.md", "chunk_id": 1}
        }
    ]
    
    with open(jsonl_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    return jsonl_path, test_data


def test_load_chunked_documents(temp_dirs, mock_embeddings, sample_jsonl_file):
    """Test loading chunked documents from JSONL file."""
    chunked_dir, chroma_dir = temp_dirs
    jsonl_path, expected_data = sample_jsonl_file
    
    creator = createChromaDB(
        embeddings=mock_embeddings,
        chunked_docs_dir=chunked_dir,
        chroma_db_dir=chroma_dir,
        collection_name="test_collection"
    )
    
    docs = creator.load_chunked_documents()
    
    assert len(docs) == 3
    assert docs[0].page_content == "This is the first chunk of text."
    assert docs[1].page_content == "This is the second chunk of text."
    assert docs[2].page_content == "This is the third chunk of text."


def test_load_chunked_documents_filters_empty_content(temp_dirs, mock_embeddings):
    """Test that empty or whitespace-only content is filtered out."""
    chunked_dir, chroma_dir = temp_dirs
    jsonl_path = chunked_dir / "test_empty.jsonl"
    
    test_data = [
        {"page_content": "Valid content", "metadata": {}},
        {"page_content": "   ", "metadata": {}},
        {"page_content": "", "metadata": {}},
        {"page_content": "Another valid content", "metadata": {}}
    ]
    
    with open(jsonl_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    creator = createChromaDB(
        embeddings=mock_embeddings,
        chunked_docs_dir=chunked_dir,
        chroma_db_dir=chroma_dir,
        collection_name="test_collection"
    )
    
    docs = creator.load_chunked_documents()
    print(docs[0].page_content)
    print(docs[0].metadata)
    
    assert len(docs) == 2
    assert all(doc.page_content.strip() for doc in docs)
