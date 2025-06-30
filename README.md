# Secure RAG-powered Chat Application

A secure and privacy-focused chat application leveraging Retrieval-Augmented Generation (RAG) with support for both public and private conversations. Built with LangChain, Gradio, and Ollama.

## 🌟 Features

- **Dual Chat Modes**: Public and Private chat interfaces
- **Document Processing**: Support for TXT, PDF, and Excel files
- **Secure Data Handling**: 
  - Encrypted vector storage
  - Metadata sanitization
  - Local LLM deployment via Ollama
- **RAG Implementation**: Enhanced responses using document context
- **Memory Management**: Conversation history with secure cleanup
- **Performance Optimized**: Multi-threading and efficient vector search

## 🔧 Prerequisites

- Python 3.8+
- Ollama installed locally
- Required Python packages (see requirements.txt)
- OpenAI API key (optional, for GPT model)

## 📦 Installation

1. Clone the repository:

  bash git clone [https://github.com/yourusername/secure-rag-chat.git](https://github.com/yourusername/secure-rag-chat.git) cd secure-rag-chat

2. Create and activate virtual environment:

  bash python -m venv venv source venv/bin/activate # For Unix venv\Scripts\activate # For Windows

3. Install dependencies:

  bash pip install -r requirements.txt

4. Install Ollama following instructions at [Ollama.ai](https://ollama.ai)

5. Set up environment variables:

  bash cp .env.example .env

## 🚀 Usage

1. Start Ollama server:  
  bash cp .env.example .env

2. Run the application:
  bash
# For public chat
python PublicChat.py
# For private chat
python PrivateChat.py

3. Access the interface:
- Open your browser to `http://localhost:7860`

## 🔐 Security Features

- **Vector Store Encryption**: All document embeddings are encrypted
- **Metadata Sanitization**: Sensitive information is masked
- **Local LLM**: Uses Ollama for private data processing
- **Memory Management**: Secure cleanup of sensitive data
- **Document Processing**: Safe handling of various file formats

## 🛠️ Configuration

Key configuration variables in `.env`:
OPENAI_API_KEY=your_api_key_here # Optional, 
for GPT model MY_FOLDER=path/to/your/documents 
MODEL_LLAMA=llama2:latest 
MODEL_EMBEDDING=all-MiniLM-L6-v2

## 📊 Performance Optimization

- Cosine similarity for faster vector search
- Configurable chunk size and overlap
- Multi-threading support
- Optimized Ollama parameters
- Efficient memory management

## 📊 Future enhancement

- Drop down list for folder input
- Detailed Excel uploader
- Add authentication and access control
- Secure memory management (

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👤 Author

- **Shahril Mohd**
- Email: mohd.shahrils@yahoo.com
- Copyright © 2025

## 🙏 Acknowledgments

- LangChain team
- Gradio team
- Ollama project
- HuggingFace community
