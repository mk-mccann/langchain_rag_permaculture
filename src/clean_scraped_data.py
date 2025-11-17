import os
import re
import json


def clean_text(text):
    # Remove excessive line breaks and whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Collapse multiple line breaks
    text = re.sub(r'[ \t]+', ' ', text)     # Collapse multiple spaces
    text = text.strip()                     # Remove leading/trailing whitespace
    return text

def split_into_sections(text):
    sections = re.split(r'\n## ', text)  # Split by Markdown-style headings
    return ["## " + section if i > 0 else section for i, section in enumerate(sections)]



# Process each file
for filename in os.listdir(input_dir):

    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as file:
        raw_text = file.read()

    # Clean the text
    cleaned_text = clean_text(raw_text)

    # Save the cleaned text
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as file:
        file.write(cleaned_text)


if __name__ == "__main__":
    input_dir = "../data/raw/scraped_pages"
    output_dir = "../processed/"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Cleaned files saved to {output_dir}/")



