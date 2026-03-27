from pydantic import BaseModel


class UploadCoursewareResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int
