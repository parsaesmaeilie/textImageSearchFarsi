Queries)
📌 Overview

This project implements a multimodal search system that allows users to search for images using Persian (Farsi) text queries.
It leverages two main components:

Translation Model – Translates Persian queries into English to align with pretrained multimodal models.

Text-Image Correlation Model – Uses a vision-language model (e.g., CLIP) to measure similarity between the translated query and candidate images.

As a result, users can input Persian text and retrieve the most relevant images with high accuracy.

🚀 Features

🔤 Persian Query Support – Seamlessly search images using Persian text.

🌍 Machine Translation Integration – Automatically converts queries to English for better model alignment.

🖼️ Image-Text Matching – Finds the most relevant images using a powerful text-image similarity model.

📊 High Performance – Provides accurate results across various datasets.

🛠️ Tech Stack

Python 3.10+

Hugging Face Transformers (translation model)

OpenAI CLIP / Similar VLM (text-image correlation)

PyTorch

NumPy, Pandas

Matplotlib / PIL (image visualization)


📂 Project Structure
```plaintext
├── notebooks/            # Jupyter notebooks for experiments and prototyping
│   └── demo.ipynb        # Example notebook to test search
├── src/                  # Core source code
│   └── search.py         # Text-image correlation & retrieval
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

⚙️ Installation

Clone the repository and install dependencies:

git clone [https://github.com/parsaesmaeilie/textImageSearchFarsi.git](https://github.com/parsaesmaeilie/textImageSearchFarsi.git)
cd your-repo
pip install -r requirements.txt

▶️ Usage

Run a search with a Persian query:

python src/search.py 
Example input:

یک گربه در حال بازی

Example output:

Query (Persian): یک گربه در حال بازی

Translation (English): A cat playing

Top-3 Retrieved Images:

📈 Results

Successfully retrieves highly relevant images given Persian text queries.

Outperforms naive keyword matching approaches.

Demonstrates the power of combining machine translation with vision-language models.

🔮 Future Work

Improve translation quality for domain-specific queries.

Add support for direct Persian embeddings (without translation).

Extend to video-text search.

👨‍💻 Author

Developed by [Your Name].
Feel free to open issues or contribute via pull requests.
