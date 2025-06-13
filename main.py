import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from celery.result import AsyncResult
from tasks import run_full_analysis

app = FastAPI(
    title="Classroom Audio Analysis Service",
    description="An API to analyze classroom audio and transcripts.",
    version="1.0.0"
)

@app.post("/analyze", status_code=202)
def start_analysis(
    transcript_file: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    """
    Accepts files, saves them temporarily, and starts the analysis task
    in the background. Returns a task ID to check for results later.
    """
    temp_dir = f"/app/analysis_output/temp_{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    transcript_path = os.path.join(temp_dir, transcript_file.filename)
    audio_path = os.path.join(temp_dir, audio_file.filename)

    with open(transcript_path, "wb") as buffer:
        shutil.copyfileobj(transcript_file.file, buffer)
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    task_id = uuid.uuid4()
    output_dir = f"/app/analysis_output/{task_id}"

    task = run_full_analysis.delay(
        transcript_path=transcript_path,
        audio_path=audio_path,
        output_dir=output_dir
    )

    return {"task_id": task.id, "message": "Analysis has been started."}


@app.get("/results/{task_id}")
def get_task_status(task_id: str):
    """
    Fetches the status and result of a background task given its ID.
    """
    task_result = AsyncResult(task_id)
    
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return result