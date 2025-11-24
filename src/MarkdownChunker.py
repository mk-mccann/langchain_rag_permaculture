import os
import re
import yaml
import json
from tqdm import tqdm
from alive_progress import alive_it
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from keybert import KeyBERT
from model2vec import StaticModel


class MarkdownChunkerWithKeywordExtraction:

    def __init__(self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        keybert_model: str = "minishlab/potion-base-8M",
        num_keywords: int = 5,
        use_maxsum: bool = True,
        use_mmr: bool = False,
        cache_dir: str | Path = "./chunked_documents"
    ):
        
        """
        Initialize enhanced markdown chunker with YAML and KeyBERT support.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            keybert_model: Model name for KeyBERT (default uses sentence-transformers)
            num_keywords: Number of keywords to extract per chunk
            use_maxsum: Use Max Sum Similarity for keyword diversity
            use_mmr: Use Maximal Marginal Relevance for keyword diversity
            cache_dir: Directory to cache processed documents
        """
    
        # Initialize KeyBERT
        self.kw_model = KeyBERT(model=StaticModel.from_pretrained(keybert_model))
        self.num_keywords = num_keywords
        self.use_maxsum = use_maxsum
        self.use_mmr = use_mmr

        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track processed files
        self.index_file = self.cache_dir / "index.json"
        self.processed_files = self._load_index()
        
        # Define headers to split on
        self.headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
        ]
        
        # Initialize markdown splitter
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        # Initialize text splitter for large sections
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )


    def _load_index(self) -> Dict:
        """Load index of processed files."""
        
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    

    def _save_index(self):
        """Save index of processed files."""
        
        with open(self.index_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    

    def _get_file_hash(self, file_path: str) -> str:
        """Get file modification time as hash."""
        
        return str(os.path.getmtime(file_path))


    def extract_frontmatter(self, markdown_content: str) -> tuple[Dict, str]:
        """
        Extract YAML frontmatter from markdown content.
        
        Args:
            markdown_content: Full markdown content with optional frontmatter
            
        Returns:
            Tuple of (frontmatter_dict, content_without_frontmatter)
        """

        # Pattern to match YAML frontmatter
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(frontmatter_pattern, markdown_content, re.DOTALL)
        
        if match:
            yaml_content = match.group(1)

            try:
                frontmatter = yaml.safe_load(yaml_content)
                # Remove frontmatter from content
                content = markdown_content[match.end():]
                return frontmatter or {}, content
            
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse YAML frontmatter: {e}")
                return {}, markdown_content
        
        return {}, markdown_content
    

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using KeyBERT.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """

        if not text or len(text.strip()) < 10:
            return []
        
        try:
            # Extract keywords with specified method
            if self.use_mmr:
                keywords = self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_mmr=True,
                    diversity=0.7,
                    top_n=self.num_keywords
                )

            elif self.use_maxsum:
                keywords = self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_maxsum=True,
                    nr_candidates=20,
                    top_n=self.num_keywords
                )
                
            else:
                keywords = self.kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=self.num_keywords
                )
            
            # Extract just the keyword strings (not scores)
            return [kw[0] for kw in keywords]
        
        except Exception as e:
            print(f"Warning: Failed to extract keywords: {e}")
            return []
        

    def chunk_markdown_file(
        self,
        markdown_content: str,
        extract_keywords: bool = True
    ) -> List[Document]:
        
        """
        Chunk markdown file with frontmatter extraction and keyword generation.
        
        Args:
            markdown_content: Full markdown content with YAML frontmatter
            extract_keywords: Whether to extract keywords using KeyBERT
            
        Returns:
            List of LangChain Document objects with enriched metadata
        """

        # Extract frontmatter
        frontmatter, content = self.extract_frontmatter(markdown_content)
        
        # Split by headers
        header_splits = self.markdown_splitter.split_text(content)
        
        # Further split large sections
        documents = self.text_splitter.split_documents(header_splits)
        
        # Enrich each document with frontmatter and keywords
        for doc in documents:
            # Add frontmatter metadata to each chunk
            doc.metadata.update(frontmatter)
            
            # Extract keywords if enabled
            if extract_keywords:
                keywords = self.extract_keywords(doc.page_content)
                doc.metadata['keywords'] = keywords
            
            # Create a hierarchical section title
            headers = [
                doc.metadata.get('header_1', ''),
                doc.metadata.get('header_2', ''),
                doc.metadata.get('header_3', ''),
                doc.metadata.get('header_4', '')
            ]
            section_path = ' > '.join(filter(None, headers))
            if section_path:
                doc.metadata['section_path'] = section_path
        
        return documents
    

    def save_documents_jsonl(self, documents: List[Document], output_file: str):
        """
        Save documents to JSON Lines format (recommended).
        One document per line - easy to append and human-readable.
        """

        output_path = self.cache_dir / output_file

        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                doc_dict = {
                    'page_content': doc.page_content,
                    'metadata': doc.metadata
                }
                f.write(json.dumps(doc_dict, ensure_ascii=False) + '\n')

        # print(f"Saved {len(documents)} documents to {output_path}")
    

    def load_documents_jsonl(self, input_file: str) -> List[Document]:
        """Load documents from JSON Lines format."""

        input_path = self.cache_dir / input_file
        documents = []
        
        if not input_path.exists():
            print(f"File not found: {input_path}")
            return documents
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc_dict = json.loads(line)
                doc = Document(
                    page_content=doc_dict['page_content'],
                    metadata=doc_dict['metadata']
                )
                documents.append(doc)
        
        # print(f"Loaded {len(documents)} documents from {input_path}")

        return documents


    def process_file_with_cache(
        self,
        file_path: str,
        force_reprocess: bool = False,
        extract_keywords: bool = True
        ) -> List[Document]:
        
        """
        Process a markdown file with caching.
        Only reprocesses if file has changed or force_reprocess=True.
        """
        
        file_path = str(Path(file_path).resolve())
        file_hash = self._get_file_hash(file_path)
        
        # Maintain one level of parent directory in cache structure
        file_path_obj = Path(file_path)
        parent_dir = file_path_obj.parent.name
        file_name = file_path_obj.stem
        cache_key = f"{parent_dir}/{file_name}"
        cache_file = f"{parent_dir}/{file_name}.jsonl"

        # Ensure cache subdirectory exists
        cache_subdir = self.cache_dir / parent_dir
        cache_subdir.mkdir(parents=True, exist_ok=True)
        
        # Check if already processed and unchanged
        if not force_reprocess and cache_key in self.processed_files:
            if self.processed_files[cache_key]['hash'] == file_hash:
                # print(f"Loading cached chunks for {file_path}")
                return self.load_documents_jsonl(cache_file)
        
        # Process the file
        # print(f"Processing {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = self.chunk_markdown_file(content, extract_keywords)
        
        # Add file path to metadata
        for doc in documents:
            doc.metadata['file_path'] = file_path
        
        # Save to cache
        self.save_documents_jsonl(documents, cache_file)
        
        # Update index
        self.processed_files[cache_key] = {
            'file_path': file_path,
            'hash': file_hash,
            'cache_file': cache_file,
            'num_chunks': len(documents),
            'processed_at': datetime.now().isoformat()
        }
        self._save_index()
        
        return documents
    

    def process_directory(
        self,
        directory: str,
        pattern: str = "**/*.md",
        force_reprocess: bool = False,
        extract_keywords: bool = True
        ) -> List[Document]:
        
        """
        Process all markdown files in a directory with intelligent caching.
        Only processes new or modified files.
        """
        
        dir_path = Path(directory)
        md_files = list(dir_path.glob(pattern))
        
        print(f"Found {len(md_files)} markdown files")
        all_documents = []
        
        for file_path in alive_it(md_files, title="Chunking markdown files"):
            try:
                documents = self.process_file_with_cache(
                    str(file_path),
                    force_reprocess,
                    extract_keywords
                )
                all_documents.extend(documents)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"\nTotal: {len(all_documents)} chunks from {len(md_files)} files")
        return all_documents
    

    # def load_all_cached_documents(self) -> List[Document]:
    #     """Load all cached documents for building vector database."""

    #     all_documents = []
        
    #     for file_info in self.processed_files.values():
    #         cache_file = file_info['cache_file']
    #         documents = self.load_documents_jsonl(cache_file)
    #         all_documents.extend(documents)
        
    #     print(f"Loaded {len(all_documents)} total cached documents")

    #     return all_documents



if __name__ == "__main__":
    import sys


    if len(sys.argv) < 1:
        print("Usage: python chunk_files.py <markdown_directory> ...")
        sys.exit(1)
    
    directory_path = sys.argv[1:]
    directory_path = Path(directory_path[0])
    
    chunker = MarkdownChunkerWithKeywordExtraction(
        num_keywords=5,
        use_maxsum=False,
        use_mmr=False,
        cache_dir=directory_path.parent / "chunked_documents"
    )
    
    documents = chunker.process_directory(directory_path, extract_keywords=True)
    
    print(f"Total documents created: {len(documents)}")
    # for doc in documents[:5]:  # Print first 5 documents as a sample
    #     print(f"Metadata: {doc.metadata}")
    #     print(f"Content Preview: {doc.page_content[:100]}...\n")
