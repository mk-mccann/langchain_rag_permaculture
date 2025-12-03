import os
import re
import yaml
import json
from alive_progress import alive_bar
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import spacy
import hashlib  
from keybert import KeyBERT
from model2vec import StaticModel      
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


class MarkdownChunkerWithKeywordExtraction:

    def __init__(self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        keybert_model: str = "minishlab/potion-base-8M",
        num_keywords: int = 5,
        split_by_headers: bool = True,
        split_by_nlp: bool = False,
        use_maxsum: bool = True,
        use_mmr: bool = False,
        cache_dir: str | Path = "./chunked_documents"
    ):
        
        """
        Initialize enhanced markdown chunker with YAML and KeyBERT support.
        Generates an index of chunked files for intelligent caching when embedding
        for incremental updates.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            keybert_model: Model name for KeyBERT (default uses sentence-transformers)
            num_keywords: Number of keywords to extract per chunk
            split_by_headers: Whether to split by markdown headers first
            split_by_nlp: Whether to use NLP-based chunking
            use_maxsum: Use Max Sum Similarity for keyword diversity
            use_mmr: Use Maximal Marginal Relevance for keyword diversity
            cache_dir: Directory to cache processed documents
        """
    
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize KeyBERT
        self.kw_model = KeyBERT(model=StaticModel.from_pretrained(keybert_model))
        self.num_keywords = num_keywords
        self.use_maxsum = use_maxsum
        self.use_mmr = use_mmr

        self.header_split = split_by_headers
        self.nlp_split = split_by_nlp

        if split_by_headers and split_by_nlp:
            self._choose_chunking_method()
        
        if not split_by_headers and not split_by_nlp:
            print("ℹ️  Using size-based chunking.")

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

        # Define separators for text splitter
        self.separators = [
            "\n\n",  # Split by paragraphs first
            "\n",    # Then by newlines
            " ",     # Then by spaces (for long sentences)
            "",      # Fallback to characters
        ]
        
        # Initialize markdown splitter
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        # Initialize text splitter for large sections
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Define NLP for nlp-based chunking - Load spaCy's English model
        self.nlp = spacy.load("en_core_web_sm")


    def _choose_chunking_method(self):
        print("\n⚠️  Cannot use both header splitting and NLP splitting simultaneously.")
        print("Please choose one splitting method:\n")
        print("  1. Header-based splitting (recommended for structured documents)")
        print("  2. NLP-based splitting (recommended for natural text)")
        print("  3. Size-based splitting (simple fallback)")
        
        while True:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            if choice == "1":
                self.header_split = True
                self.nlp_split = False
                print("✓ Using header-based splitting")
                break
            elif choice == "2":
                self.header_split = False
                self.nlp_split = True
                print("✓ Using NLP-based splitting")
                break
            elif choice == "3":
                self.header_split = False
                self.nlp_split = False
                print("✓ Using size-based splitting")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")


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
    

    def _get_file_stats(self, file_path: str) -> tuple[float, int]:
        """Return (mtime, size) for a file."""
        st = os.stat(file_path)
        return st.st_mtime, st.st_size


    def _sha256_file(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file's contents (streamed)."""
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()


    def _useNLPChunking(self, content: str) -> List[Document]:
        """
        Chunk text using NLP sentence boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """

        # Process the content with spaCy
        doc = self.nlp(content)

        # Group sentences into coherent chunks
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_size = self.chunk_size  # Target chunk size in characters

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)

            # Add sentence to current chunk if it fits
            if current_length + sent_length <= max_chunk_size:
                current_chunk.append(sent_text)
                current_length += sent_length
            else:
                # Finalize the current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent_text]
                current_length = sent_length

        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Post-process to merge short chunks and handle lists
        final_chunks = []
        for chunk in chunks:
            # Detect lists and merge them with the next chunk if split
            if re.match(r'^\s*(\d+\.|-|\*)', chunk) and len(chunk) < max_chunk_size * 0.5:
                # Merge with next chunk if it's a list and too short
                if chunks.index(chunk) < len(chunks) - 1:
                    next_chunk = chunks[chunks.index(chunk) + 1]
                    if re.match(r'^\s*(\d+\.|-|\*)', next_chunk):
                        chunk += "\n" + next_chunk
                        chunks.remove(next_chunk)
            final_chunks.append(chunk)

        # Create documents with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={}
            )
            for chunk in final_chunks
        ]

        return documents


    def _sizeChunking(self, content: str) -> List[Document]:
        # Directly split by size
        chunks = self.text_splitter.split_text(content)

        # Post-process to merge list items
        final_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            # Detect if the chunk starts a list
            if re.match(r'^\s*(\d+\.|-|\*)', chunk):
                # Merge with next chunks until the list ends
                merged_chunk = chunk
                j = i + 1
                while j < len(chunks) and re.match(r'^\s*(\d+\.|-|\*)', chunks[j]):
                    merged_chunk += "\n" + chunks[j]
                    j += 1
                final_chunks.append(merged_chunk)
                i = j
            else:
                final_chunks.append(chunk)
                i += 1

        # Create documents with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={}
            )
            for chunk in final_chunks
        ]

        return documents
    

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
                # Parse YAML with custom date handling
                frontmatter = yaml.safe_load(yaml_content)

                # Convert any date objects to ISO format strings for JSON serialization
                if frontmatter:
                    for key, value in frontmatter.items():
                        if key in ['access_date', 'date']:
                            frontmatter[key] = str(value)
                        if isinstance(value, str):
                            # Preserve special characters as strings
                            frontmatter[key] = value.encode('utf-8').decode('unicode_escape')
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
        
        # Split by headers, if enabled
        if self.header_split:
            header_splits = self.markdown_splitter.split_text(content)
        
            # Further split large sections by size
            documents = self.text_splitter.split_documents(header_splits)

        elif self.nlp_split:
            # Use NLP-based chunking
            documents = self._useNLPChunking(content)

        else:
            # Direct size-based chunking
            documents = self._sizeChunking(content)
        
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
        
        return documents


    def process_file_with_cache(
        self,
        file_path: Path|str,
        force_reprocess: bool = False,
        extract_keywords: bool = True
        ) -> List[Document]:
        
        """
        Process a markdown file with caching.
        Only reprocesses if file has changed or force_reprocess=True.

        Args:
            file_path (Path|str): Path to markdown file
            force_reprocess (bool): Whether to force reprocessing
            extract_keywords (bool): Whether to extract keywords
        Returns:
            List of LangChain Documents
        """
        
        file_path = str(Path(file_path).resolve())
        
        # Maintain one level of parent directory in cache structure
        file_path_obj = Path(file_path)
        parent_dir = file_path_obj.parent.name
        file_name = file_path_obj.stem
        cache_key = f"{parent_dir}/{file_name}"
        cache_file = f"{parent_dir}/{file_name}.jsonl"

        # Ensure cache subdirectory exists
        cache_subdir = self.cache_dir / parent_dir
        cache_subdir.mkdir(parents=True, exist_ok=True)
        
        # Check if already processed and unchanged using mtime/size/content_hash
        if not force_reprocess and cache_key in self.processed_files:
            cached = self.processed_files[cache_key]
            cached_mtime = cached.get('mtime')
            cached_size = cached.get('size')
            cached_hash = cached.get('content_hash')
            
            if cached_mtime is not None and cached_size is not None and cached_hash:
                # Fast check: compare mtime and size
                try:
                    current_mtime, current_size = self._get_file_stats(file_path)
                    if current_mtime == cached_mtime and current_size == cached_size:
                        # Unchanged: skip reprocessing
                        return self.load_documents_jsonl(cache_file)
                    # Changed: verify with content hash
                    current_hash = self._sha256_file(file_path)
                    if current_hash == cached_hash:
                        # False positive from mtime/size; content unchanged
                        return self.load_documents_jsonl(cache_file)
                except Exception:
                    # File issue; reprocess
                    pass
        
        # Process the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = self.chunk_markdown_file(content, extract_keywords)
        
        # Add file path to metadata
        for doc in documents:
            doc.metadata['file_path'] = file_path
        
        # Save to cache
        self.save_documents_jsonl(documents, cache_file)
        
        # Update index with mtime, size, and content hash
        mtime, size = self._get_file_stats(file_path)
        content_hash = self._sha256_file(file_path)
        
        # Also record JSONL mtime for diagnostics/fallback
        jsonl_mtime = (self.cache_dir / cache_file).stat().st_mtime if (self.cache_dir / cache_file).exists() else None
        self.processed_files[cache_key] = {
            'file_path': file_path,
            'mtime': mtime,
            'size': size,
            'content_hash': content_hash,
            'cache_file': cache_file,
            'jsonl_mtime': jsonl_mtime,
            'num_chunks': len(documents),
            'processed_at': datetime.now().isoformat()
        }
        self._save_index()
        
        return documents
    

    def process_directory(
        self,
        directory: Path|str,
        force_reprocess: bool = False,
        extract_keywords: bool = True,
        max_workers: int|None = None
        ) -> List[Document]:
        
        """
        Process all markdown files in a directory with intelligent caching.
        Only processes new or modified files. Multiple files are processed in parallel.

        Args:
            directory (Path|str): Directory containing markdown files
            force_reprocess (bool): Whether to force reprocessing of all files
            extract_keywords (bool): Whether to extract keywords
            max_workers (int|None): Maximum number of parallel workers (default: 1)

        Returns:
            List of LangChain Documents
        """
        
        dir_path = Path(directory)
        md_files = list(dir_path.glob("**/*.md"))
        
        print(f"Found {len(md_files)} markdown files")
        all_documents = []

        # Parallelize file processing with a thread pool while preserving progress bar        
        # Determine worker count: user-specified or default to single-threaded
        if max_workers is None:
            max_workers = 1
        else:
            max_workers = min((os.cpu_count() or 4), int(max_workers))

        def _process(path: Path) -> tuple[Path, List[Document]]:
            try:
                docs = self.process_file_with_cache(
                    str(path),
                    force_reprocess,
                    extract_keywords
                )
                return path, docs
            except Exception as e:
                print(f"Error processing {path}: {e}")
                return path, []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_process, p): p for p in md_files}
            processed = 0

            with alive_bar(len(md_files), title="Chunking markdown files", dual_line=True) as bar:

                for future in as_completed(future_map):
                    _, docs = future.result()

                    if docs:
                        all_documents.extend(docs)

                    processed += 1
                    bar.text = f"Processed {processed}/{len(md_files)} files"
                    bar()

        print(f"\nTotal: {len(all_documents)} chunks from {len(md_files)} files")
        return all_documents
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-directory",
        type=str,
        default="../data/raw",
        required=True,
        help="Directory containing markdown files to chunk"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers for chunking"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of all files regardless of cache"
    )
    parser.add_argument(
        "--no-keywords",
        action="store_true",
        default=False,
        help="Disable keyword extraction"
    )

    args = parser.parse_args()
    directory_path = Path(args.data_directory)
    
    chunker = MarkdownChunkerWithKeywordExtraction(
        num_keywords=5,
        use_maxsum=False,
        use_mmr=False,
        split_by_nlp=False,
        split_by_headers=False,
        cache_dir=directory_path.parents[1] / "chunked_documents"
    )
    
    documents = chunker.process_directory(
        directory_path,
        extract_keywords=args.no_keywords,
        force_reprocess=args.force_reprocess,
        max_workers=args.max_workers
    )
    
    print(f"Total documents created: {len(documents)}")
