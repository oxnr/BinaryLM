import json
import os
import shutil

# Ensure output directories exist
output_dir_web = os.path.join('web', 'public', 'notebooks')
output_dir_main = os.path.join('notebooks')
os.makedirs(output_dir_web, exist_ok=True)
os.makedirs(output_dir_main, exist_ok=True)

# Basic sample notebook
sample_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Sample Jupyter Notebook\n",
                "\n",
                "This is a basic notebook to test the notebook viewer component."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# A simple Python cell\n",
                "print(\"Hello, world!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Markdown Example\n",
                "\n",
                "Here's some markdown content with:\n",
                "- Bullet points\n",
                "- **Bold text**\n",
                "- *Italic text*\n",
                "\n",
                "And a code block:\n",
                "```python\n",
                "def hello():\n",
                "    print(\"Hello from a code block\")\n",
                "```"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Simple LM notebook (just a placeholder, very minimal)
lm_scratch_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Building a Language Model from Scratch\n",
                "\n",
                "This notebook demonstrates how to build a simple transformer-based language model using PyTorch."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import torch.nn as nn\n",
                "import numpy as np\n",
                "\n",
                "print(\"PyTorch version:\", torch.__version__)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Transformers notebook (just a placeholder, very minimal)
transformers_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Building LMs with the Transformers Library\n",
                "\n",
                "This notebook demonstrates how to use the Hugging Face Transformers library to work with pre-trained language models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
                "\n",
                "# Load a pre-trained model\n",
                "tokenizer = AutoTokenizer.from_pretrained(\"gpt2\")\n",
                "model = AutoModelForCausalLM.from_pretrained(\"gpt2\")\n",
                "\n",
                "print(\"Model loaded successfully!\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# First check if we need to make simple empty files
notebooks = {
    "sample_notebook.ipynb": sample_notebook,
    "Build_LM_From_Scratch.ipynb": lm_scratch_notebook,
    "Building_with_Transformers.ipynb": transformers_notebook
}

# Write the basic notebooks to files
for filename, notebook in notebooks.items():
    output_path_web = os.path.join(output_dir_web, filename)
    output_path_main = os.path.join(output_dir_main, filename)
    
    with open(output_path_web, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    
    with open(output_path_main, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Created {output_path_web}")
    print(f"Created {output_path_main}")

# Copy the 5-minute demo notebook if it exists in the main directory
demo_notebook_main = os.path.join(output_dir_main, "LLM_5min_Demo.ipynb")
demo_notebook_web = os.path.join(output_dir_web, "LLM_5min_Demo.ipynb")

# If the demo notebook exists in main directory but not in web/public, copy it
if os.path.exists(demo_notebook_main) and not os.path.exists(demo_notebook_web):
    shutil.copy2(demo_notebook_main, demo_notebook_web)
    print(f"Copied LLM_5min_Demo.ipynb to {demo_notebook_web}")
elif not os.path.exists(demo_notebook_main) and os.path.exists(demo_notebook_web):
    shutil.copy2(demo_notebook_web, demo_notebook_main)
    print(f"Copied LLM_5min_Demo.ipynb to {demo_notebook_main}")
elif not os.path.exists(demo_notebook_main) and not os.path.exists(demo_notebook_web):
    print("Warning: LLM_5min_Demo.ipynb not found in either location!")

print("All sample notebooks created successfully!") 