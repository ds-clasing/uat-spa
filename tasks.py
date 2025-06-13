import os
import shutil
from classroom_analyzer import ClassroomAudioAnalyzer
from celery_config import celery_app

@celery_app.task(bind=True)
def run_full_analysis(self, transcript_path: str, audio_path: str, output_dir: str):
    """
    A Celery task that runs the full audio analysis.
    The `self` argument is a Celery convention.
    """
    try:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        analyzer = ClassroomAudioAnalyzer(
            transcript_file=transcript_path,
            audio_file=audio_path,
            gemini_api_key=gemini_key,
            output_dir_base=output_dir
        )
        
        reports = analyzer.run_analysis()

        os.remove(transcript_path)
        os.remove(audio_path)

        return {"status": "Complete", "reports_generated": len(reports) if reports else 0}
    except Exception as e:
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        print(f"Task failed: {e}")
        raise