# LangGraph Agent for Document Processing

An intelligent AI agent built with LangGraph and DeepSeek for working with documents and articles. The agent can process uploaded documents, answer questions, search for information, and generate summaries.

## 🚀 Features

- **Document Upload & Processing**: Support for PDF and text files (up to 5 MB)
- **Semantic Search**: Vector search over uploaded documents using Pinecone
- **Web Search**: Find similar documents via DuckDuckGo
- **Summary Generation**: Automatic creation of document summaries
- **Question Generation**: Create relevant questions based on document content
- **Intelligent Routing**: Automatic selection of appropriate tools for task execution

## 🏗️ Architecture

Built with:
- **LangGraph** - for creating agent state graphs
- **LangChain** - framework for working with LLMs
- **DeepSeek Chat** - language model for agent and text processing
- **FastAPI** - web server with REST API
- **Pinecone** - vector database for semantic search
- **PyMuPDF (fitz)** - PDF text extraction

## 📋 Requirements

- Python 3.8+
- DeepSeek API key
- Pinecone API key

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langgraph-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your API keys:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## 🚀 Getting Started

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The server will be available at: `http://localhost:8000`

## 📡 API Endpoints

### 1. Upload File
```
POST /files/
```
**Parameters:**
- `file`: File to upload (PDF or text, max 5 MB)

**Response:**
```json
{
  "status": "ok",
  "length": 3000,
  "user_id": "uuid-string"
}
```

### 2. Agent Request
```
POST /agent-request/
```
**Parameters:**
- `user_input`: User's query text
- `user_id`: User ID (received when uploading a file)

**Response:**
```json
{
  "response": "Agent's response to the query"
}
```

## 🛠️ Available Agent Tools

The agent automatically selects the appropriate tool based on the request:

1. **browsing** - Search DuckDuckGo for similar documents or up-to-date information
2. **ingesting** - Split and store documents in a vector database
3. **retrieving** - Semantic search over stored documents
4. **text_agent** - Generate summaries and/or questions based on the document
5. **help_tool** - Information about the agent's capabilities

## 📂 Project Structure

```
langgraph-agent/
├── main.py                 # FastAPI application entry point
├── agent.py                # LangGraph agent graph definition
├── tools.py                # Tools for the agent
├── doc_loader.py           # Document loading and processing utilities
├── config.py               # Configuration settings
├── requirements.txt        # Project dependencies
├── routes/
│   ├── __init__.py
│   ├── agent_requests.py   # Endpoint for agent requests
│   └── files.py            # Endpoint for file uploads
└── data/                   # Sample documents
```

## 💡 Usage Examples

1. **Upload a document:**
```bash
curl -X POST "http://localhost:8000/files/" \
  -F "file=@document.pdf"
```

2. **Ask a question about the document:**
```bash
curl -X POST "http://localhost:8000/agent-request/" \
  -F "user_input=What are the main topics discussed in the document?" \
  -F "user_id=your-user-id"
```

3. **Get a summary:**
```bash
curl -X POST "http://localhost:8000/agent-request/" \
  -F "user_input=Create a brief summary of this document" \
  -F "user_id=your-user-id"
```

## 🔄 Agent Workflow

The agent uses a LangGraph state graph:

1. **START** → Receives user request
2. **Agent Node** → LLM analyzes the request and selects tools
3. **Tool Node** → Selected tools are executed
4. **Loop** → Agent can make multiple tool calls
5. **END** → Returns final response

## ⚙️ Configuration

- **MAX_FILE_SIZE**: 5 MB (default)
- **File cache TTL**: 3600 seconds (1 hour)
- **Maximum cached files**: 100
- **Chunk size for vectorization**: 500 characters
- **Chunk overlap**: 100 characters

## 📝 Logging

The project uses standard Python logging. Logs are enabled at application startup and track:
- User file uploads
- Tool calls
- LLM errors

## 🔒 Limitations

- Maximum file size: 5 MB
- Only the first 3000 characters of uploaded documents are processed
- Files are cached for 1 hour
- Supported formats: PDF, text files

## 🤝 Contributing

Contributions are welcome! Feel free to open issues and pull requests.

## 📄 License

[Specify your license]

## 👤 Author

[Specify author information]
