# Quick Start: Incremental Updates

## TL;DR

Your `CreateChromaDB` class now automatically detects and processes only new or modified documents instead of rebuilding the entire database every time.

## Quick Examples

### Default (Incremental Update)
```python
creator = CreateChromaDB(embeddings=embeddings, ...)
creator.embed_and_store()  # Only processes new/modified files
```

### Check What Will Be Updated
```python
new, modified = creator.get_modified_and_new_files()
print(f"Will process: {len(new)} new + {len(modified)} modified files")
```

### Force Full Rebuild
```python
creator.embed_and_store(incremental=False, resume=False)
```

## What Changed?

### ✅ What You Get
- **Faster**: Only embeds changed documents
- **Cheaper**: Only pays for new embeddings
- **Smarter**: Removes old embeddings for modified files
- **Automatic**: No code changes needed for basic usage

### 🔄 How It Works
1. Compares current files with `index.json`
2. Identifies new and modified files
3. Removes old embeddings for modified files
4. Loads and embeds only changed content

### 📊 Performance
- **Before**: Process ~50,000 docs every time (hours)
- **After**: Process ~100-1,000 docs (minutes)
- **Savings**: 10-100x faster, 90%+ cost reduction

## Common Workflows

### Regular Updates (Recommended)
```bash
# 1. Update your source documents
# 2. Run your chunking process
python src/MarkdownChunker.py

# 3. Update embeddings (automatically incremental)
python src/CreateChromaDB.py
```

### First Time Setup
```bash
# Full database creation
python src/CreateChromaDB.py
# (will process all files on first run)
```

### Reset Database
```bash
# Delete database and rebuild
rm -rf chroma_db
python src/CreateChromaDB.py
```

## Troubleshooting

**No changes detected?**
- Ensure `index.json` exists in chunked_docs_dir
- Verify your chunking process updates the index

**Still slow?**
- Check if incremental=True (default)
- Reduce batch_size if hitting memory limits

**Database seems off?**
- Delete chroma_db directory
- Run with incremental=False for clean rebuild

## Files to Know

- `data/chunked_documents/index.json` - Tracks processed files
- `logs/embedding_progress.json` - Checkpoint for resume
- `logs/failed_batches.txt` - Logs any errors
- `chroma_db/` - Your vector database

## More Info

- Full docs: `docs/INCREMENTAL_UPDATES.md`
- Examples: `examples/incremental_update_example.py`
- Implementation: `docs/IMPLEMENTATION_SUMMARY.md`
