import json
import os
from pathlib import Path
from alive_progress import alive_it


def update_md_frontmatter():
    """Update frontmatter of all markdown files in data/raw/ with sources"""
    # Process all markdown files in data/raw/
    data_raw_path = Path("../../data/raw/scraped_pages/")
    
    for md_file in alive_it(data_raw_path.glob("**/*.md"), title="Updating frontmatter..."):
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find matching source from config
        # Get the source from the parent directory
        source = md_file.parent.name
        
        if source:
            # Define desired frontmatter key order
            key_order = ['source', 'title', 'author', 'url', 'access_date', 'date', 'license', 'description', 'keywords']
            
            # Check if frontmatter exists
            if content.startswith('---'):
                # Update existing frontmatter
                parts = content.split('---', 2)
                
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    body = parts[2]
                    
                    # Parse existing frontmatter into a dict
                    lines = frontmatter.strip().split('\n')
                    frontmatter_dict = {}
                    
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)

                            # Do a bunch of formatting to make the YAML frontmatter easier to parse later on
                            key = key.strip()
                            value = value.strip()
                            
                            # If value is empty, set to empty string
                            if not value:
                                value = ""
                            
                            # if value is in quotes, remove them
                            if value.startswith('"') or value.startswith("'"):
                                value = value[1:] 
                            if value.endswith('"') or value.endswith("'"):
                                value = value[:-1] 

                            # If there is a colon in the value for any key, replace with dash
                            if key in ['title', 'description'] and ':' in value:
                                value = value.replace(':', ' -')

                            # if I fucked up and removed the colon from the url, add it back
                            if key == 'url' and value.startswith('https'):
                                value = value.replace('https -', 'https:')
                            
                            frontmatter_dict[key] = value
                    
                    # Update or add source field
                    frontmatter_dict["source"] = source
                    
                    # Rebuild frontmatter in desired order
                    new_lines = []
                    for key in key_order:
                        if key in frontmatter_dict:
                            new_lines.append(f"{key}: {frontmatter_dict[key]}")
                    
                    # Add any remaining keys not in key_order
                    for key, value in frontmatter_dict.items():
                        if key not in key_order:
                            new_lines.append(f"{key}: {value}")
                    
                    new_content = f"---\n{chr(10).join(new_lines)}\n---{body}"
                else:
                    # Add frontmatter
                        new_content = f"---\nsource: {source}\n---\n{content}"
            else:
                # Add new frontmatter
                new_content = f"---\nsource: {source}\n---\n{content}"
            
            # Write back to file
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # print(f"Updated {md_file.source} with source: {source}")


if __name__ == "__main__":
    update_md_frontmatter()
