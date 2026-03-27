from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.schemas import UploadCoursewareResponse
from app.vectorstore import get_vectorstore


app = FastAPI(title="ClassMate Backend")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload_courseware", response_model=UploadCoursewareResponse)
async def upload_courseware(file: UploadFile = File(...)) -> UploadCoursewareResponse:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    safe_name = Path(file.filename).name
    file_id = uuid4().hex
    saved_path = UPLOAD_DIR / f"{file_id}_{safe_name}"

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    saved_path.write_bytes(file_bytes)

    try:
        loader = PyPDFLoader(str(saved_path))
        documents = loader.load()
        if not documents:
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        split_docs = splitter.split_documents(documents)

        for i, doc in enumerate(split_docs):
            doc.metadata["source_file"] = safe_name
            doc.metadata["chunk_index"] = i

        vectorstore = get_vectorstore()
        vectorstore.add_documents(split_docs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index PDF: {exc}",
        ) from exc

    return UploadCoursewareResponse(
        message="Courseware uploaded and indexed successfully.",
        filename=safe_name,
        chunks_indexed=len(split_docs),
    )
