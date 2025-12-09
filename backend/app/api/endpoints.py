from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import os
import numpy as np
from ..services.audio_processor import AudioProcessor
from ..services.noise_classifier import NoiseClassifier
from ..services.cough_classifier import CoughClassifier
from ..services.explainable_ai import ExplainableAI
from ..config import config
import traceback
import base64
import fastapi   # <-- add this line

router = APIRouter()

# Initialize services
audio_processor = AudioProcessor()
noise_classifier = NoiseClassifier()
cough_classifier = CoughClassifier()


# =========================================================
#   MAIN ANALYSIS ENDPOINT (handles file + base64)
# =========================================================
@router.post("/analyze-audio")
async def analyze_audio(
    uploaded_file: Optional[UploadFile] = File(None),
    audio_data: Optional[str] = Form(None)
):
    print("\n===== /analyze-audio HIT =====")
    # SAFE print – no .filename crash
    print("FILE:", getattr(uploaded_file, 'filename', 'live-recording'))
    print("AUDIO_DATA:", "Yes" if audio_data else "No")
    if isinstance(uploaded_file, fastapi.params.File):
        uploaded_file = None   # treat as empty

    try:
        # ---------------- CASE 1: USER UPLOADED A FILE ----------------
        if uploaded_file is not None:
            print("Reading uploaded file...")
            file_bytes = await uploaded_file.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file_bytes)
                audio_path = tmp.name

        # ---------------- CASE 2: USER SENT BASE64 AUDIO --------------
        elif audio_data is not None:
            print("Decoding base64 audio...")

            # Remove base64 header if present
            if "," in audio_data:
                audio_data = audio_data.split(",")[1]

            # Fix missing padding
            missing_padding = len(audio_data) % 4
            if missing_padding != 0:
                audio_data += "=" * (4 - missing_padding)

            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 audio: {str(e)}"
                )

            # ---- write a valid WAV file ----
            if len(audio_bytes) < 100:   # too small
                raise HTTPException(status_code=400, detail="Audio too short (< 0.1 s)")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                # write a proper WAV header + PCM data
                import soundfile as sf
                pcm = np.frombuffer(audio_bytes, dtype=np.int16)
                sf.write(tmp.name, pcm, samplerate=22050)
                audio_path = tmp.name

            print("Decoded audio size:", len(audio_bytes))

        else:
            raise HTTPException(status_code=400, detail="No audio data provided")

        # ---------------- PROCESSING ---------------------------------
        print("Processing audio file...")
        result = process_audio_file(audio_path)

        os.unlink(audio_path)
        print("Temporary audio deleted.")

        return JSONResponse(content=result)

    except Exception as e:
        print("🔥 ERROR in /analyze-audio:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
#   FILE UPLOAD ENDPOINT
# =========================================================
@router.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    print("\n===== /analyze-file HIT =====")
    return await analyze_audio(uploaded_file=file)


# =========================================================
#   RECORDING ENDPOINT (BASE64)
# =========================================================
@router.post("/analyze-recording")
async def analyze_recording(audio_data: str = Form(...)):
    print("\n===== /analyze-recording HIT =====")
    print("Received recorded audio (base64 length):", len(audio_data))
    return await analyze_audio(audio_data=audio_data)


# =========================================================
#   MODEL INFO
# =========================================================
@router.get("/model-info")
async def get_model_info():
    print("\n===== /model-info HIT =====")
    return {
        "noise_classifier": {
            "model_path": str(config.NOISE_CLASSIFIER_PATH),
            "exists": config.NOISE_CLASSIFIER_PATH.exists()
        },
        "cough_classifier": {
            "model_path": str(config.COUGH_CLASSIFIER_PATH),
            "exists": config.COUGH_CLASSIFIER_PATH.exists()
        },
        "audio_config": {
            "sample_rate": config.SAMPLE_RATE,
            "duration": config.DURATION,
            "n_mfcc": config.N_MFCC
        }
    }


# =========================================================
#   PROCESSOR PIPELINE
# =========================================================
def process_audio_file(audio_path: str) -> dict:
    print("\n===== PROCESSING AUDIO FILE =====")
    print("Audio Path:", audio_path)

    try:
        audio, sr = audio_processor.load_audio(audio_path)
        processed_audio = audio_processor.preprocess_audio(audio)
        features = audio_processor.extract_features(processed_audio)

        noise_result = noise_classifier.predict(features)

        # If not cough → attempt cough segment extraction
        if not noise_result["is_cough"]:
            cough_segments = audio_processor.detect_cough_segments(processed_audio)

            if not cough_segments:
                return {
                    "status": "error",
                    "message": "No cough detected. Provide clearer recording.",
                    "noise_classification": noise_result
                }

            best_segment = max(cough_segments, key=lambda x: x["confidence"])
            features = audio_processor.extract_features(best_segment["segment"])
            noise_result = noise_classifier.predict(features)

        cough_result = cough_classifier.predict(features)

        explainable_ai = ExplainableAI(cough_classifier.model, cough_classifier.class_names)
        explanation = explainable_ai.explain_prediction(features, cough_result)

        return {
            "status": "success",
            "noise_classification": noise_result,
            "cough_classification": cough_result,
            "explanation": explanation,
            "audio_info": {
                "duration": len(processed_audio) / sr,
                "sample_rate": sr,
                "features_extracted": list(features.keys())
            }
        }

    except Exception as e:
        print("🔥 ERROR in process_audio_file:", str(e))
        traceback.print_exc()
        return {"status": "error", "message": f"Error processing audio: {str(e)}"}


# =========================================================
#   TRAIN MODELS
# =========================================================
@router.post("/train-models")
async def train_models():
    print("\n===== /train-models HIT =====")
    try:
        from ..training.train_noise_classifier import train_noise_classifier
        from ..training.train_cough_classifier import train_cough_classifier

        noise_results = train_noise_classifier()
        cough_results = train_cough_classifier()

        return {
            "status": "success",
            "noise_classifier_training": noise_results,
            "cough_classifier_training": cough_results
        }

    except Exception as e:
        print("🔥 ERROR in /train-models:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")