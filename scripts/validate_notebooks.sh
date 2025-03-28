#!/bin/bash

# Script to validate notebook JSON and ensure proper formatting

set -e
echo "Validating notebook files..."

NOTEBOOKS_DIR="notebooks"
PUBLIC_DIR="web/public/notebooks"

# Create public notebooks directory if it doesn't exist
mkdir -p "$PUBLIC_DIR"

# Ensure the sample notebook is copied
echo "Copying sample notebook..."
cp "web/public/notebooks/sample_notebook.ipynb" "$PUBLIC_DIR/sample_notebook.ipynb" 2>/dev/null || true

# Process each notebook file
for notebook in "$NOTEBOOKS_DIR"/*.ipynb; do
    filename=$(basename "$notebook")
    echo "Processing $filename..."
    
    # Validate JSON formatting
    if ! jq empty "$notebook" 2>/dev/null; then
        echo "Error: $filename is not valid JSON. Attempting to fix..."
        # Try to fix common JSON issues
        python -c "import json; f=open('$notebook', 'r'); data=json.load(f); f.close(); f=open('$notebook', 'w'); json.dump(data, f, indent=2); f.close();" 2>/dev/null || echo "Failed to fix $filename"
    fi
    
    # Copy to public directory (even if validation failed, to enable debugging)
    cp "$notebook" "$PUBLIC_DIR/$filename"
    echo "Copied $filename to public directory"
    
    # Final validation check
    if jq empty "$PUBLIC_DIR/$filename" 2>/dev/null; then
        echo "✅ $filename is valid JSON"
    else
        echo "⚠️ Warning: $filename in public directory may not be valid JSON"
    fi
done

echo "Notebook validation complete." 