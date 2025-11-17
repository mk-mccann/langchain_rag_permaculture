import os
from pathlib import Path

def remove_empty_files(directory: str = "data/raw/scraped_pages"):
    """Remove empty files from the specified directory."""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory {directory} does not exist.")
        return
    
    removed_count = 0
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.stat().st_size == 0:
            print(f"Removing empty file: {file_path}")
            file_path.unlink()
            removed_count += 1
    
    print(f"Removed {removed_count} empty file(s).")

def remove_non_txt_files(directory: str = "data/raw/scraped_pages"):
    """This is a utility function to remove non-txt files from the specified directory.
    Specifically, it removes any files that have a double extension or are not .txt files."""

    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory {directory} does not exist.")
        return
    
    removed_count = 0
    for file_path in dir_path.iterdir():
        if file_path.is_file() and (file_path.suffix != '.txt' or len(file_path.suffixes) > 1):
            print(f"Removing non-txt file: {file_path}")
            file_path.unlink()
            removed_count += 1
    
    print(f"Removed {removed_count} non-txt file(s).")



if __name__ == "__main__":
    remove_empty_files("../data/raw/scraped_pages")
    remove_non_txt_files("../data/raw/scraped_pages")
