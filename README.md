# ClassMate Backend (FastAPI + LangChain + ChromaDB)

## Features

- FastAPI service
- `POST /upload_courseware` for PDF courseware upload
- LangChain PDF loading and chunk splitting
- ChromaDB local vector storage initialization
- OpenAI embeddings with `text-embedding-3-small`

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

3. Start server:

```bash
uvicorn app.main:app --reload
```

## API

### POST `/upload_courseware`

- Form field: `file` (PDF)
- Behavior:
  - saves PDF to `uploads/`
  - extracts text via `PyPDFLoader`
  - splits text to chunks
  - indexes into local ChromaDB at `chroma_db/`
