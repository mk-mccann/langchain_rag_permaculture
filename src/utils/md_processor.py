"""
Markdown File Processing Utilities

This module provides functions for cleaning and processing markdown files,
both individually and in batch (directory) mode. It combines functionality
for removing unwanted files, cleaning boilerplate content, and updating
frontmatter metadata.

Functions:
    - clean_single_file: Clean a single markdown file
    - clean_directory: Batch process a directory of markdown files
    - update_single_file_frontmatter: Update frontmatter for a single file
    - update_directory_frontmatter: Batch update frontmatter for all files in directory
    - process_single_file: Full processing pipeline for a single file
    - process_directory: Full processing pipeline for a directory
"""


import re
from os import cpu_count
from typing import Optional, Dict, Tuple, Any

from pathlib import Path
from alive_progress import alive_bar
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configuration constants
DISALLOWED_FILE_TYPES = [
    '.png', '.jpg', '.jpeg', '.gif', '.pdf', 
    '.docx', '.zip', '.fcstd', '.stl', '.kdenlive',
    '.mp4', '.mp3', '.avi', '.mov', '.svg', '.skp',
    '.exe', '.dmg', '.iso', '.tar', '.gz', '.rar', '.7z', '.csv',
    '.xlsx', '.pptx', '.ini', '.sys', '.dll', '.dxf', '.odt', 
    '.ods', '.odp', '.epub', '.mobi', '.dae', '.fbx', '.3ds',
    '.ino', '.stp'
]

DISALLOWED_PAGE_TYPES = [
    'File:', 'Schematic:', 'Category:', 'Special:', 
    'Template:', 'one-community-welcomes'
]

BOILERPLATE_INDICATORS = [
    "Navigation menu",
    "Contribute to this page",
    "###### WHO WE ARE",
    "###### WHO IS ONE COMMUNITY",
    "Retrieved from",
]

FRONTMATTER_KEY_ORDER = [
    'source', 'title', 'author', 'url', 'access_date', 
    'date', 'license', 'description', 'keywords'
]


# ============================================================================
# SINGLE FILE OPERATIONS
# ============================================================================

def should_remove_file(file_path: Path) -> Tuple[bool, str]:
    """
    Determine if a file should be removed based on various criteria.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple of (should_remove: bool, reason: str)
    """
    # Check if empty
    if file_path.stat().st_size == 0:
        return True, "empty file"
    
    # Check disallowed page types
    for page_type in DISALLOWED_PAGE_TYPES:
        if page_type in file_path.name:
            return True, f"disallowed page type: {page_type}"
    
    # Check file extensions and keywords
    for file_type in DISALLOWED_FILE_TYPES:
        if file_type in file_path.name.lower():
            return True, f"disallowed file type: {file_type}"
    
    # Check tag files
    if file_path.name[:3] == "tag":
        return True, "tag file"
    
    return False, ""


def remove_boilerplate(content: str) -> Tuple[str, bool]:
    """
    Remove boilerplate text from markdown content.
    
    Args:
        content: The markdown content to clean
        
    Returns:
        Tuple of (cleaned_content: str, was_modified: bool)
    """
    original_content = content
    
    for indicator in BOILERPLATE_INDICATORS:
        index = content.find(indicator)
        if index != -1:
            content = content[:index].strip()
    
    return content, content != original_content


def remove_duplicate_lines(content: str) -> Tuple[str, bool]:
    """
    Remove duplicate lines from markdown content.
    
    Args:
        content: The markdown content to process
        
    Returns:
        Tuple of (cleaned_content: str, was_modified: bool)
    """
    lines = content.splitlines()
    unique_lines = []
    seen = set()
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line not in seen:
            unique_lines.append(line)
            seen.add(stripped_line)
    
    cleaned_content = "\n".join(unique_lines)
    return cleaned_content, cleaned_content != content


# def remove_duplicated_headings(md_content: str) -> str:


def standardize_headings(md_content: str) -> str:
    # Split frontmatter and content
    parts = re.split(r'^---\n.*?\n---\n', md_content, flags=re.DOTALL, maxsplit=1)
    frontmatter = f"---\n{parts[1].strip()}\n---\n" if len(parts) > 1 else ""
    content = parts[-1] if len(parts) > 1 else md_content

    # Standardize headers in content only
    content = re.sub(r'^(.+)\n=+', r'# \1', content, flags=re.MULTILINE)
    content = re.sub(r'^(.+)\n-+', r'## \1', content, flags=re.MULTILINE)

    # Recombine frontmatter and standardized content
    return frontmatter + content


def fix_frontmatter(md_content: str) -> Tuple[str, str]:
    # Split the content into lines
    lines = md_content.split('\n')

    # Find the index of the last line in the frontmatter (contains ':')
    last_frontmatter_index = 0
    for i, line in enumerate(lines):
        # Check for URLs that are NOT part of the frontmatter key (break on content URLs)
        if ("https:" in line or "http:" in line) and not ("url: https:" in line or "url: http:" in line):
            break  # Stop at first URL in content (not frontmatter)
        # Special case: url field may contain "url: https:" or "url: http:"
        elif ("url: https:" in line) or ("url: http:" in line):
            last_frontmatter_index = i
        elif (':' in line) or ("---" in line):
            last_frontmatter_index = i
        else:
            break  # Stop at the first line without ':' after frontmatter lines

    # Reconstruct the frontmatter and content
    frontmatter_lines = lines[:last_frontmatter_index + 1]

    # Remove any lines that are "---" (YAML delimiters)
    frontmatter_lines = [line for line in frontmatter_lines if line.strip() != '---']

    # Remove any duplicate keys in frontmatter
    seen_keys = set()
    unique_frontmatter_lines = []
    for line in frontmatter_lines:
        if "##" in line:
            continue  # Skip comment lines
        key = line.split(':', 1)[0].strip()
        if key not in seen_keys:
            unique_frontmatter_lines.append(line)
            seen_keys.add(key)

    frontmatter = '---\n' + '\n'.join(unique_frontmatter_lines) + '\n---\n'
    content = '\n'.join(lines[last_frontmatter_index + 1:])

    # Standardize headers in the content
    content = re.sub(r'^(.+)\n=+', r'# \1', content, flags=re.MULTILINE)
    content = re.sub(r'^(.+)\n-+', r'## \1', content, flags=re.MULTILINE)

    # Combine the fixed frontmatter and standardized content
    return frontmatter, content


def rename_wiki_file(file_path: Path) -> Optional[Path]:
    """
    Rename files that start with 'wiki' by removing the prefix.
    
    Args:
        file_path: Path to the file to potentially rename
        
    Returns:
        New path if renamed, None otherwise
    """
    if file_path.name.startswith("wiki"):
        new_name = file_path.name[4:]
        new_path = file_path.with_name(new_name)
        file_path.rename(new_path)
        return new_path
    return None


def parse_frontmatter(content: str) -> Tuple[Dict[str, str], str]:
    """
    Parse YAML frontmatter from markdown content.
    
    Args:
        content: Markdown content with potential frontmatter
        
    Returns:
        Tuple of (frontmatter_dict, body_content)
    """
    frontmatter_dict = {}
    body = content
    
    if content.startswith('---'):
        parts = content.split('---', 2)
        
        if len(parts) >= 3:
            frontmatter = parts[1]
            body = parts[2]
            
            lines = frontmatter.strip().split('\n')
            
            for line in lines:
                if "##" in line:
                    continue  # Skip comment lines
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Clean up the value
                    if not value:
                        value = ""
                    
                    # Remove quotes
                    if value.startswith('"') or value.startswith("'"):
                        value = value[1:] 
                    if value.endswith('"') or value.endswith("'"):
                        value = value[:-1]
                    
                    # Replace colons in certain fields
                    if key in ['title', 'description'] and ':' in value:
                        value = value.replace(':', ' -')
                    
                    # Fix URL colons
                    if key == 'url' and value.startswith('https -'):
                        value = value.replace('https -', 'https:')
                    
                    frontmatter_dict[key] = value
    
    return frontmatter_dict, body


def build_frontmatter(frontmatter_dict: Dict[str, str]) -> str:
    """
    Build YAML frontmatter string from dictionary.
    
    Args:
        frontmatter_dict: Dictionary of frontmatter key-value pairs
        
    Returns:
        Formatted frontmatter string
    """
    lines = []
    
    # Add keys in preferred order
    for key in FRONTMATTER_KEY_ORDER:
        if key in frontmatter_dict:
            lines.append(f"{key}: {frontmatter_dict[key]}")
    
    # Add any remaining keys not in order
    for key, value in frontmatter_dict.items():
        if key not in FRONTMATTER_KEY_ORDER:
            lines.append(f"{key}: {value}")
    
    return "\n".join(lines)


def clean_single_file(
    file_path: Path | str,
    remove_boilerplate_text: bool = True,
    remove_duplicates: bool = True,
    rename_wiki: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Clean a single markdown file.
    
    Args:
        file_path: Path to the markdown file
        remove_boilerplate_text: Whether to remove boilerplate content
        remove_duplicates: Whether to remove duplicate lines
        rename_wiki: Whether to rename files starting with 'wiki'
        verbose: Print detailed information
        
    Returns:
        Dictionary with processing results
    """
    file_path = Path(file_path)
    results = {
        'file': str(file_path),
        'removed': False,
        'reason': None,
        'modified': False,
        'changes': []
    }
    
    if not file_path.exists():
        results['error'] = 'File does not exist'
        return results
    
    # Check if file should be removed
    should_remove, reason = should_remove_file(file_path)
    if should_remove:
        if verbose:
            print(f"Removing {file_path}: {reason}")
        file_path.unlink()
        results['removed'] = True
        results['reason'] = reason
        return results
    
    # Only process markdown files
    if file_path.suffix != '.md':
        results['skipped'] = True
        results['reason'] = 'Not a markdown file'
        return results
    
    # Read file content
    try:
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        results['error'] = str(e)
        return results
    
    original_content = content

    # Fix frontmatter
    frontmatter, content = fix_frontmatter(content)
        
    # Remove boilerplate
    if remove_boilerplate_text:
        content, modified = remove_boilerplate(content)
        if modified:
            results['changes'].append('removed_boilerplate')
    
    # Remove duplicate lines
    if remove_duplicates:
        content, modified = remove_duplicate_lines(content)
        if modified:
            results['changes'].append('removed_duplicates')

    # Standardize headings
    content_before_headings = content
    content = standardize_headings(content)
    if content != content_before_headings:
        results['changes'].append('standardized_headings')
    
    # Rebuild full content
    content = frontmatter + content

    # Write back if modified
    if content != original_content:
        try:
            with file_path.open('w', encoding='utf-8') as f:
                f.write(content)
            results['modified'] = True
            if verbose:
                print(f"Cleaned {file_path}: {', '.join(results['changes'])}")
        except Exception as e:
            results['error'] = str(e)
            return results
    
    # Rename if needed
    if rename_wiki:
        new_path = rename_wiki_file(file_path)
        if new_path:
            results['renamed'] = True
            results['new_path'] = str(new_path)
            results['changes'].append('renamed_wiki')
            if verbose:
                print(f"Renamed {file_path} to {new_path}")
    
    return results


def update_single_file_frontmatter(
    file_path: Path | str,
    source: Optional[str] = None,
    additional_metadata: Optional[Dict[str, str]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Update frontmatter for a single markdown file.
    
    Args:
        file_path: Path to the markdown file
        source: Source identifier (defaults to parent directory name)
        additional_metadata: Additional key-value pairs to add to frontmatter
        verbose: Print detailed information
        
    Returns:
        Dictionary with processing results
    """
    file_path = Path(file_path)
    results = {
        'file': str(file_path),
        'modified': False,
        'added_fields': [],
        'updated_fields': []
    }
    
    if not file_path.exists():
        results['error'] = 'File does not exist'
        return results
    
    if file_path.suffix != '.md':
        results['skipped'] = True
        results['reason'] = 'Not a markdown file'
        return results
    
    # Determine source
    if source is None:
        source = file_path.parent.name
    
    # Read file
    try:
        with file_path.open('r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        results['error'] = str(e)
        return results
    
    # Parse existing frontmatter
    frontmatter_dict, body = parse_frontmatter(content)
    
    # Track what's new vs updated
    if 'source' in frontmatter_dict:
        if frontmatter_dict['source'] != source:
            results['updated_fields'].append('source')
    else:
        results['added_fields'].append('source')
    
    # Update source
    frontmatter_dict['source'] = source.replace('_', ' ')

    # There are older files where 'name' was used instead of 'source'
    if 'name' in frontmatter_dict:
        del frontmatter_dict['name']

    # Occasionally the description field may have colons in it, and we want to fix that because it
    # can break parsers since it thinks it's a new key. So we replace colons with dashes in description.
    for key in frontmatter_dict.keys():
        if key not in FRONTMATTER_KEY_ORDER:
            # This is probably the offending field
            trailing = " -".join([key, frontmatter_dict[key]])

            if "description" in frontmatter_dict.keys():
                frontmatter_dict["description"] = frontmatter_dict["description"] + trailing
            else:
                print(trailing + " could not be fixed automatically.")

    # Add additional metadata
    if additional_metadata:
        for key, value in additional_metadata.items():
            if key in frontmatter_dict:
                if frontmatter_dict[key] != value:
                    results['updated_fields'].append(key)
            else:
                results['added_fields'].append(key)
            frontmatter_dict[key] = value
    
    # Build new content
    frontmatter_str = build_frontmatter(frontmatter_dict)
    new_content = f"---\n{frontmatter_str}\n---{body}"
    
    # Write back
    if new_content != content:
        try:
            with file_path.open('w', encoding='utf-8') as f:
                f.write(new_content)
            results['modified'] = True
            if verbose:
                print(f"Updated frontmatter for {file_path}")
        except Exception as e:
            results['error'] = str(e)
            return results
    
    return results


def process_single_file(
    file_path: Path | str,
    clean: bool = True,
    update_frontmatter: bool = True,
    source: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Full processing pipeline for a single file (clean + update frontmatter).
    
    Args:
        file_path: Path to the markdown file
        clean: Whether to clean the file
        update_frontmatter: Whether to update frontmatter
        source: Source identifier for frontmatter
        verbose: Print detailed information
        
    Returns:
        Dictionary with combined processing results
    """
    file_path = Path(file_path)
    results = {'file': str(file_path)}
    
    if clean:
        clean_results = clean_single_file(file_path, verbose=verbose)
        results["clean"] = clean_results
        
        # If file was removed, stop here
        if clean_results.get('removed'):
            return results
        
        # Update file_path if renamed
        if clean_results.get('renamed'):
            file_path = Path(clean_results['new_path'])
    
    if update_frontmatter and file_path.exists():
        frontmatter_results = update_single_file_frontmatter(
            file_path, source=source, verbose=verbose
        )
        results['frontmatter'] = frontmatter_results
    
    return results


# ============================================================================
# BATCH/DIRECTORY OPERATIONS
# ============================================================================

def clean_directory(
    directory: Path | str,
    recursive: bool = True,
    workers: Optional[int] = 1,
    remove_boilerplate_text: bool = True,
    remove_duplicates: bool = True,
    rename_wiki: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Clean all markdown files in a directory (multithreaded).
    
    Args:
        directory: Path to directory to process
        recursive: Process subdirectories recursively
        workers: Number of parallel worker threads (default: CPU count * 2)
        remove_boilerplate_text: Whether to remove boilerplate content
        remove_duplicates: Whether to remove duplicate lines
        rename_wiki: Whether to rename wiki files
        verbose: Print detailed information
        show_progress: Show progress bar
        
    Returns:
        Dictionary with summary statistics
    """

    dir_path = Path(directory)
    if not dir_path.exists():
        return {'error': f'Directory {directory} does not exist'}
    
    stats = {
        'total_files': 0,
        'removed': 0,
        'modified': 0,
        'renamed': 0,
        'skipped': 0,
        'errors': 0
    }
    
    # Collect all files (filter to files only)
    if recursive:
        files = [f for f in dir_path.glob('**/*') if f.is_file()]
    else:
        files = [f for f in dir_path.iterdir() if f.is_file()]
    
    stats['total_files'] = len(files)
    if not files:
        return stats

    # Worker callable
    def _process(path: Path) -> Dict[str, Any]:
        return clean_single_file(
            path,
            remove_boilerplate_text=remove_boilerplate_text,
            remove_duplicates=remove_duplicates,
            rename_wiki=rename_wiki,
            verbose=verbose
        )

    # Thread pool size: use I/O-bound friendly count
    if not workers or workers < 1:
        max_workers = max(4, (cpu_count() or 4) * 2)
    else:
        max_workers = workers

    # Process files with progress bar
    with alive_bar(len(files), title="Cleaning files") as bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_process, f): f for f in files}
            for future in as_completed(future_to_path):
                result = future.result()
                if result.get('removed'):
                    stats['removed'] += 1
                elif result.get('modified'):
                    stats['modified'] += 1
                if result.get('renamed'):
                    stats['renamed'] += 1
                if result.get('skipped'):
                    stats['skipped'] += 1
                if result.get('error'):
                    stats['errors'] += 1
                bar()

    return stats


def update_directory_frontmatter(
    directory: Path | str,
    workers: Optional[int] = 1,
    recursive: bool = True,
    source: Optional[str] = None,
    use_parent_as_source: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Update frontmatter for all markdown files in a directory (multithreaded).
    
    Args:
        directory: Path to directory to process
        workers: Number of parallel worker threads (default: CPU count * 2)
        recursive: Process subdirectories recursively
        source: Source identifier (if None and use_parent_as_source=True, uses parent dir)
        use_parent_as_source: Use parent directory name as source if source not provided
        verbose: Print detailed information
        
    Returns:
        Dictionary with summary statistics
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        return {'error': f'Directory {directory} does not exist'}

    stats = {
        'total_files': 0,
        'modified': 0,
        'skipped': 0,
        'errors': 0
    }

    # Collect markdown files
    if recursive:
        md_files = list(dir_path.glob('**/*.md'))
    else:
        md_files = list(dir_path.glob('*.md'))

    stats['total_files'] = len(md_files)
    if not md_files:
        return stats

    def _process(file_path: Path) -> Dict[str, Any]:
        # Determine source for this file
        file_source = source
        if file_source is None and use_parent_as_source:
            file_source = file_path.parent.name

        return update_single_file_frontmatter(
            file_path,
            source=file_source,
            verbose=verbose
        )

    # Thread pool size: use I/O-bound friendly count
    if not workers or workers < 1:
        max_workers = max(4, (cpu_count() or 4) * 2)
    else:
        max_workers = workers

    # Process files with progress bar
    with alive_bar(len(md_files), title="Updating frontmatter") as bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_process, f): f for f in md_files}
            for future in as_completed(future_to_path):
                result = future.result()
                if result.get('modified'):
                    stats['modified'] += 1
                if result.get('skipped'):
                    stats['skipped'] += 1
                if result.get('error'):
                    stats['errors'] += 1
                bar()

    return stats


def process_directory(
    directory: Path | str,
    recursive: bool = True,
    workers: Optional[int] = 1,
    clean: bool = True,
    update_frontmatter: bool = True,
    source: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Full processing pipeline for a directory (clean + update frontmatter).
    
    Args:
        directory: Path to directory to process
        recursive: Process subdirectories recursively
        workers: Number of parallel worker threads (default: CPU count * 2)
        clean: Whether to clean files
        update_frontmatter: Whether to update frontmatter
        source: Source identifier for frontmatter
        verbose: Print detailed information
        
    Returns:
        Dictionary with combined summary statistics
    """
    results = {'directory': str(directory)}
    
    if clean:
        if verbose:
            print("Cleaning files...")
        clean_stats = clean_directory(
            directory,
            recursive=recursive,
            verbose=verbose,
            workers=workers,
        )
        results['clean'] = clean_stats
    
    if update_frontmatter:
        if verbose:
            print("Updating frontmatter...")
        frontmatter_stats = update_directory_frontmatter(
            directory,
            recursive=recursive,
            source=source,
            verbose=verbose,
            workers=workers,
        )
        results['frontmatter'] = frontmatter_stats
    
    return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def print_results(results: Dict[str, Any], operation: str = "Processing"):
    """
    Pretty print processing results.
    
    Args:
        results: Results dictionary from processing functions
        operation: Name of the operation for display
    """
    print(f"\n{operation} Results:")
    print("=" * 60)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Handle single file results
    if 'file' in results:
        print(f"File: {results['file']}")
        if results.get('removed'):
            print(f"  ✓ Removed: {results.get('reason', 'N/A')}")
        elif results.get('modified'):
            print(f"  ✓ Modified: {', '.join(results.get('changes', []))}")
        elif results.get('skipped'):
            print(f"  ⊘ Skipped: {results.get('reason', 'N/A')}")
    
    # Handle directory results
    if 'clean' in results:
        clean = results['clean']
        print("\nCleaning:")
        print(f"  Total files: {clean.get('total_files', 0)}")
        print(f"  Removed: {clean.get('removed', 0)}")
        print(f"  Modified: {clean.get('modified', 0)}")
        print(f"  Renamed: {clean.get('renamed', 0)}")
        print(f"  Skipped: {clean.get('skipped', 0)}")
        print(f"  Errors: {clean.get('errors', 0)}")
    
    if 'frontmatter' in results:
        fm = results['frontmatter']
        print("\nFrontmatter Updates:")
        print(f"  Total files: {fm.get('total_files', 0)}")
        print(f"  Modified: {fm.get('modified', 0)}")
        print(f"  Skipped: {fm.get('skipped', 0)}")
        print(f"  Errors: {fm.get('errors', 0)}")
    
    print("=" * 60)


# ============================================================================
# MAIN / EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Example usage
    print("Markdown Processor Utility")
    print("=" * 60)
    
    # Default directory
    default_dir = Path("../../data/raw/md_scraped_pages")
    
    # Process based on command line args
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        
        if target.is_file():
            # Process single file
            print(f"\nProcessing single file: {target}")
            results = process_single_file(target, verbose=True, update_frontmatter=False)
            print_results(results, "Single File Processing")
        
        elif target.is_dir():
            # Process directory
            print(f"\nProcessing directory: {target}")
            results = process_directory(target, verbose=False, update_frontmatter=False, workers=3)
            print_results(results, "Directory Processing")
        
        else:
            print(f"Error: {target} does not exist")
    
    else:
        # Default: process default directory
        if default_dir.exists():
            print(f"\nProcessing default directory: {default_dir}")
            results = process_directory(default_dir, verbose=False, update_frontmatter=True, workers=3)
            print_results(results, "Directory Processing")
        else:
            print(f"Default directory does not exist: {default_dir}")
            print("\nUsage:")
            print(f"  python {Path(__file__).name} <file_or_directory>")
