# RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that provides accurate and context-aware responses by leveraging document retrieval and language models. This project includes a FastAPI backend and a modern web interface.

## Features

- **Document Retrieval**: Load and process documents from various sources
- **Vector Database**: Uses ChromaDB for efficient similarity search
- **OpenAI Integration**: Leverages OpenAI's language models for generating responses
- **Email Support**: Integrated email support for raising tickets
- **Web Interface**: Modern and responsive web interface using Next.js

## Prerequisites

- Python 3.10 or higher
- Node.js 16.x or higher (for frontend)
- Redis server
- OpenAI API key
- SMTP server credentials (for email support)

## Installation

### Backend Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-chatbot/backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv rag_venv
   # On Windows:
   rag_venv\Scripts\activate
   # On Unix or MacOS:
   # source rag_venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the backend directory with the following variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   REDIS_HOST=localhost
   REDIS_PORT=6379
   
   # Email configuration (optional)
   MAIL_USERNAME=your_email@example.com
   MAIL_PASSWORD=your_email_password
   MAIL_FROM=your_email@example.com
   MAIL_PORT=587
   MAIL_SERVER=smtp.example.com
   MAIL_STARTTLS=True
   MAIL_SSL_TLS=False
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Create a `.env.local` file in the frontend directory:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

## Running the Application

### Start Redis Server
Make sure Redis is running on your system. You can start it with:

```bash
# On Linux/Mac
redis-server

# On Windows (if installed via Chocolatey)
redis-server
```

### Start the Backend

From the backend directory:

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Start the Frontend

From the frontend directory:

```bash
npm run dev
```

The web interface will be available at `http://localhost:3000`

## API Endpoints

- `POST /api/ask`: Chat with the RAG model
  ```json
  {
    "query": "Your question here",
    "session_id": "unique-session-id"
  }
  ```

- `POST /api/raise-ticket`: Raise a support ticket
  ```json
  {
    "query": "Your support request"
  }
  ```

- `POST /api/clear-redis`: Clear all chat history (for development)

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── vector_store/        # Vector store implementation
│   └── loaders/             # Document loaders
└── frontend/               # Next.js frontend
    ├── public/             # Static files
    ├── src/
    │   ├── pages/         # Next.js pages
    │   └── components/     # React components
    └── package.json        # Frontend dependencies
```

## Environment Variables

### Backend
- `OPENAI_API_KEY`: Your OpenAI API key
- `REDIS_HOST`: Redis server host (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `MAIL_*`: Email configuration for support tickets
- `SUPPORT_EMAIL`: Email address to send support tickets to
- `SMTP_HOST`: SMTP server host
- `SMTP_PORT`: SMTP server port
- `SMTP_USER`: SMTP server username
- `SMTP_PASSWORD`: SMTP server password

### Frontend
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: http://localhost:8000)

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
