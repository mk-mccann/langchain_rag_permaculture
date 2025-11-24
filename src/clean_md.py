from pathlib import Path


def clean_directory(directory: Path | str = "data/raw/scraped_pages"):
    """Remove empty files, non-md files, and boilerplate text in one pass."""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory {directory} does not exist.")
        return
    
    disallowed_file_types = ['.png', '.jpg', '.jpeg', '.gif', '.pdf', 
                            '.docx', '.zip', '.fcstd', '.stl', '.kdenlive',
                            '.mp4', '.mp3', '.avi', '.mov', '.svg', '.skp', '.stl',
                            '.exe', '.dmg', '.iso', '.tar', '.gz', '.rar', '.7z', '.csv',
                            '.xlsx', '.pptx', '.ini', '.sys', '.dll', '.dxf', '.odt', 
                            '.ods', '.odp', '.epub', '.mobi', '.dae', '.fbx', '.3ds', '.dxf',
                            '.ino', '.stp']
    
    disallowed_page_types = ['File:', 'Schematic:', 'Category:', 'Special:', 'Template:', 'one-community-welcomes']
    
    boilerplate_indicators = [
        "Navigation menu",
        "Contribute to this page",
        "###### WHO WE ARE",
    ]
    
    removed_count = 0
    cleaned_count = 0
    
    for file_path in dir_path.iterdir():
        if file_path.is_dir():
            clean_directory(file_path)
        
        # Remove empty files
        if file_path.stat().st_size == 0:
            print(f"Removing empty file: {file_path}")
            file_path.unlink()
            removed_count += 1
            continue
        
        # Remove files with disallowed page types or file extensions
        if any(page_type in file_path.name for page_type in disallowed_page_types):
            file_path.unlink()
            removed_count += 1
            continue
        
        # Remove some files based on keywords in the filename
        if any(file_type in file_path.name.lower() for file_type in disallowed_file_types) or file_path.name[:3] == "tag":
            file_path.unlink()
            removed_count += 1
            continue
        
        # Remove boilerplate from markdown files
        if file_path.suffix == ".md":
            with file_path.open("r", encoding="utf-8") as file:
                content = file.read()
            
            original_content = content
            
            for indicator in boilerplate_indicators:
                index = content.find(indicator)
                if index != -1:
                    content = content[:index].strip()
            
            if content != original_content:
                with file_path.open("w", encoding="utf-8") as file:
                    file.write(content)
                cleaned_count += 1

        # Rename files where the first 4 characters are 'wiki'
        if file_path.name.startswith("wiki"):
            new_name = file_path.name[4:]
            new_path = file_path.with_name(new_name)
            file_path.rename(new_path)
        
    
    print(f"Removed {removed_count} file(s).")
    print(f"Cleaned boilerplate from {cleaned_count} markdown file(s).")


if __name__ == "__main__":
    clean_directory("../data/raw/scraped_pages")
    print("Cleanup completed.")
