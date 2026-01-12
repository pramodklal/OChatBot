# 🤖 Multi-Files Chatbot using OpenAI - RAG System

A powerful Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that allows users to upload multiple document formats (PDF, DOCX, TXT) and ask questions based on the content using OpenAI's GPT models.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)

---

## 📋 Solution Brief

This application implements a sophisticated RAG (Retrieval-Augmented Generation) system that enables users to:

- **Upload Multiple Document Types**: Support for PDF, DOCX, and TXT files
- **Intelligent Document Processing**: Automatic text extraction and chunking
- **Semantic Search**: FAISS-powered vector similarity search
- **AI-Powered Responses**: Multiple OpenAI GPT model options (GPT-4, GPT-4-Turbo, GPT-3.5-Turbo)
- **Conversational Interface**: Streamlit-based chat UI with message history
- **Error Handling**: Rate limiting protection with exponential backoff

### Key Features

✅ **Multi-format Document Support** - PDF, DOCX, TXT  
✅ **Multiple AI Models** - Choose from GPT-4, GPT-4-Turbo, or GPT-3.5-Turbo  
✅ **Vector Similarity Search** - FAISS for efficient document retrieval  
✅ **Conversational Interface** - Chat-style interaction with history  
✅ **Rate Limit Protection** - Automatic retry with exponential backoff  
✅ **Persistent Storage** - Local FAISS index for embeddings  

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│              Streamlit Chat UI + File Upload                    │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DOCUMENT PROCESSING LAYER                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   PDF       │  │    DOCX     │  │    TXT      │            │
│  │  Extractor  │  │  Extractor  │  │  Reader     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                          │                                       │
│                          ▼                                       │
│                  ┌─────────────────┐                            │
│                  │  Text Combiner  │                            │
│                  └─────────────────┘                            │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TEXT CHUNKING LAYER                         │
│         RecursiveCharacterTextSplitter                          │
│         (chunk_size: 10000, overlap: 1000)                      │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                EMBEDDING & VECTOR STORE LAYER                   │
│  ┌──────────────────────┐      ┌────────────────────┐          │
│  │  OpenAI Embeddings   │ ───► │   FAISS Vector DB  │          │
│  └──────────────────────┘      └────────────────────┘          │
│                                  (Stored: faiss_index/)         │
└─────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING LAYER                       │
│  User Question → Embedding → Similarity Search → Context        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM PROCESSING LAYER                       │
│              OpenAI ChatGPT (GPT-4/4-Turbo/3.5)                 │
│                  Context + Question → Answer                     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE GENERATION                          │
│            Streaming Output + Chat History Storage              │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

1. **Document Upload Phase**:
   ```
   Upload Files → Extract Text → Combine → Chunk → Generate Embeddings → Store in FAISS
   ```

2. **Query Phase**:
   ```
   User Question → Embed Query → Search FAISS → Retrieve Context → Generate Prompt → LLM → Response
   ```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | Streamlit | Web interface and file upload |
| **Document Parsers** | PyPDF2, python-docx | Extract text from files |
| **Text Splitter** | LangChain RecursiveCharacterTextSplitter | Break text into chunks |
| **Embeddings** | OpenAI Embeddings API | Convert text to vectors |
| **Vector Store** | FAISS (Facebook AI Similarity Search) | Store and search embeddings |
| **LLM** | OpenAI GPT-4/4-Turbo/3.5-Turbo | Generate answers |
| **Orchestration** | LangChain | Chain components together |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- OpenAI API Key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG_Gemini_BOT
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running Locally

```bash
streamlit run rag.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Step 1: Upload Documents
1. Click on the sidebar file uploader
2. Select one or multiple files (PDF, DOCX, or TXT)
3. Click "Submit & Process"
4. Wait for processing to complete

### Step 2: Select Model
Choose your preferred OpenAI model from the dropdown:
- **GPT-4**: Most capable, best for complex queries
- **GPT-4-Turbo**: Faster responses, good balance
- **GPT-3.5-Turbo**: Fastest, cost-effective

### Step 3: Ask Questions
Type your question in the chat input and press Enter. The bot will:
1. Search for relevant content in your uploaded documents
2. Generate a contextual answer using the selected GPT model
3. Display the response in the chat interface

### Step 4: Chat History
- All messages are stored in the session
- Use "Clear Chat History" button to reset

---

## 🔧 Configuration

### Text Chunking Parameters

```python
chunk_size = 10000      # Characters per chunk
chunk_overlap = 1000    # Overlap between chunks
```

### Model Settings

```python
temperature = 0.3       # Lower = more focused, Higher = more creative
```

### Rate Limiting

```python
max_retries = 3         # Maximum retry attempts
retry_delay = 40        # Initial delay in seconds
                        # Exponential backoff: 40s → 80s → 160s
```

---

## 📂 Project Structure

```
RAG_Gemini_BOT/
├── rag.py                      # Main application file
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── faiss_index/               # Vector store (auto-created)
│   └── index.faiss           # FAISS index file
├── Architecture.drawio        # System architecture diagram
├── dfd.drawio                # Data flow diagram
└── README.md                  # This file
```

---

## 🔑 Key Functions

### Document Processing

- **`get_pdf_text(pdf_file)`**: Extract text from PDF files using PyPDF2
- **`get_word_text(docx_file)`**: Extract text from DOCX files
- **`read_text_file(txt_file)`**: Read text from TXT files
- **`combine_text(text_list)`**: Merge text from multiple files

### Text Processing

- **`get_text_chunks(text)`**: Split text into manageable chunks with overlap
- **`get_vector_store(chunks)`**: Generate embeddings and store in FAISS

### Query Processing

- **`user_input(user_question, modelname)`**: Process user queries and generate responses
- **`get_conversational_chain(modelname)`**: Initialize the OpenAI chat model

### UI Management

- **`clear_chat_history()`**: Reset chat session
- **`main()`**: Streamlit application entry point

---

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `rag.py` as the main file

3. **Add Secrets**
   In Streamlit Cloud dashboard → App Settings → Secrets:
   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   ```

4. **Deploy**
   Click "Deploy" and wait for the app to start

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Programming language |
| **Streamlit** | Latest | Web framework |
| **LangChain** | Latest | LLM orchestration |
| **OpenAI** | Latest | GPT models API |
| **FAISS** | 1.7.4+ | Vector similarity search |
| **PyPDF2** | 3.0.0+ | PDF text extraction |
| **python-docx** | 1.0.0+ | DOCX text extraction |

---

## ⚠️ Error Handling

The application includes robust error handling:

- **Rate Limiting**: Automatic detection and retry with exponential backoff
- **API Errors**: Graceful error messages for quota/connection issues
- **File Processing**: Validation and error reporting for corrupt files
- **Session Management**: Proper state management to prevent crashes

---

## 📊 Performance Considerations

- **Chunk Size**: Larger chunks provide more context but slower processing
- **Overlap**: Ensures context continuity across chunk boundaries
- **Model Selection**: Balance between quality (GPT-4) and speed/cost (GPT-3.5)
- **FAISS Index**: Stored locally for fast retrieval after initial processing

---

## 🔒 Security Notes

- ⚠️ Never commit `.env` file to version control
- ⚠️ Use Streamlit Cloud secrets for deployment
- ⚠️ Keep your OpenAI API key secure
- ⚠️ Set appropriate usage limits on your OpenAI account

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: `OpenAI API Key Error`
- **Solution**: Check `.env` file exists and contains valid API key

**Issue**: `Rate Limit Exceeded`
- **Solution**: Wait for retry or upgrade OpenAI plan

**Issue**: `FAISS Index Not Found`
- **Solution**: Upload and process documents first before asking questions

---

## 📝 Future Enhancements

- [ ] Support for additional file formats (CSV, Excel, Images with OCR)
- [ ] Multi-language support
- [ ] Document source citations in responses
- [ ] Advanced filtering and search options
- [ ] User authentication and document management
- [ ] Integration with other LLM providers (Anthropic, Google)
- [ ] Agentic RAG capabilities with tool usage

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Design & Developed by Code Insights @pramodklal**

---

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [Your contact information]

---

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain for the orchestration framework
- Facebook AI for FAISS
- Streamlit for the amazing web framework
- The open-source community

---

**Built with ❤️ for the AI community**

