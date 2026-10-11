from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "signals.db"

DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Audio Signal Processing Platform",
    description="Real-time audio signal synthesis and analysis platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database with signals table"""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                dominant_frequency REAL,
                rms REAL,
                sample_rate INTEGER,
                sample_count INTEGER,
                waveform TEXT,
                payload TEXT
            )
            """
        )
        conn.commit()
    except Exception as e:
        print(f"Database init error: {e}")
    finally:
        conn.close()


init_db()


def dominant_frequency(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Calculate dominant frequency using FFT"""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < 2:
        return 0.0, 0.0

    samples = samples - np.mean(samples)
    window = np.hanning(samples.size)
    signal = samples * window

    fft_values = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    magnitudes = np.abs(fft_values)

    if magnitudes.size <= 1:
        return 0.0, 0.0

    idx = int(np.argmax(magnitudes[1:]) + 1)
    return float(freqs[idx]), float(magnitudes[idx])


def generate_waveform(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 44100,
    waveform: str = "sine",
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate waveform signal"""
    total_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    if waveform == "sine":
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
    elif waveform == "square":
        signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == "triangle":
        signal = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
    elif waveform == "saw":
        signal = amplitude * (2 * (frequency * t % 1) - 1)
    else:
        signal = amplitude * np.sin(2 * np.pi * frequency * t)

    return signal.astype(np.float32)


def store_signal_record(
    source: str,
    signal_type: str,
    dominant_frequency: float,
    rms: float,
    sample_rate: int,
    sample_count: int,
    waveform: str | None = None,
    payload: Any = None,
) -> Dict[str, Any]:
    """Store signal record in database"""
    created_at = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO signals (
                created_at, source, signal_type, dominant_frequency, rms,
                sample_rate, sample_count, waveform, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                source,
                signal_type,
                dominant_frequency,
                rms,
                sample_rate,
                sample_count,
                waveform,
                json.dumps(payload) if payload is not None else None,
            ),
        )
        conn.commit()
        signal_id = cur.lastrowid
    except Exception as e:
        print(f"Store record error: {e}")
        signal_id = 0
    finally:
        conn.close()

    return {
        "id": signal_id,
        "created_at": created_at,
        "source": source,
        "signal_type": signal_type,
        "dominant_frequency": dominant_frequency,
        "rms": rms,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "waveform": waveform,
    }


@app.get("/")
async def index() -> FileResponse:
    """Serve main HTML page"""
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "audio-signal-processing-platform",
        "version": "1.0.0"
    }


@app.get("/api/history")
async def get_history(limit: int = 25) -> List[Dict[str, Any]]:
    """Get signal history from database"""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM signals
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except Exception as e:
        print(f"History query error: {e}")
        return []
    finally:
        conn.close()

    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "source": row["source"],
            "signal_type": row["signal_type"],
            "dominant_frequency": row["dominant_frequency"],
            "rms": row["rms"],
            "sample_rate": row["sample_rate"],
            "sample_count": row["sample_count"],
            "waveform": row["waveform"],
            "payload": json.loads(row["payload"]) if row["payload"] else None,
        }
        for row in rows
    ]


@app.post("/api/synthesize")
async def synthesize_signal(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 44100,
    waveform: str = "sine",
    amplitude: float = 0.5,
) -> Dict[str, Any]:
    """Synthesize and store a signal"""
    signal = generate_waveform(
        frequency=frequency,
        duration=duration,
        sample_rate=sample_rate,
        waveform=waveform,
        amplitude=amplitude,
    )
    dominant_hz, dominant_mag = dominant_frequency(signal, sample_rate)
    rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0

    record = store_signal_record(
        source="api",
        signal_type="synthetic",
        dominant_frequency=dominant_hz,
        rms=rms,
        sample_rate=sample_rate,
        sample_count=int(signal.size),
        waveform=waveform,
        payload={
            "frequency_hz": frequency,
            "duration_seconds": duration,
            "amplitude": amplitude,
        },
    )

    return {
        "status": "stored",
        "record": record,
        "signal": {
            "frequency_hz": frequency,
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "waveform": waveform,
            "amplitude": amplitude,
            "samples": int(signal.size),
            "dominant_frequency_hz": dominant_hz,
            "dominant_magnitude": dominant_mag,
            "rms": rms,
        },
    }


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyze uploaded audio file"""
    try:
        contents = await file.read()
        
        try:
            import soundfile as sf
            import io
            buffer = io.BytesIO(contents)
            audio, sr = sf.read(buffer, dtype="float32", always_2d=False)
        except Exception:
            arr = np.frombuffer(contents, dtype=np.uint8)
            audio = np.asarray(arr, dtype=np.float32) / 255.0
            sr = 44100

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        dominant_hz, dominant_mag = dominant_frequency(audio, sr)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

        record = store_signal_record(
            source="upload",
            signal_type="captured",
            dominant_frequency=dominant_hz,
            rms=rms,
            sample_rate=int(sr),
            sample_count=int(audio.size),
            payload={"filename": file.filename},
        )

        return {
            "status": "stored",
            "record": record,
            "dominant_frequency_hz": dominant_hz,
            "dominant_magnitude": dominant_mag,
            "rms": rms,
            "samples": int(audio.size),
        }
    except Exception as exc:
        return {"error": str(exc), "status": "error"}


@app.websocket("/ws/audio")
async def audio_ws(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming"""
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)
            samples = np.asarray(payload.get("samples", []), dtype=np.float32)
            sample_rate = int(payload.get("sampleRate", 44100))

            dominant_hz, dominant_mag = dominant_frequency(samples, sample_rate)
            rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0

            record = store_signal_record(
                source="websocket",
                signal_type="live",
                dominant_frequency=dominant_hz,
                rms=rms,
                sample_rate=sample_rate,
                sample_count=int(samples.size),
                waveform=payload.get("waveform"),
                payload={
                    "message_type": payload.get("type", "mic"),
                    "record_count": len(samples),
                },
            )

            await websocket.send_json(
                {
                    "id": record["id"],
                    "dominantFrequencyHz": dominant_hz,
                    "dominantMagnitude": dominant_mag,
                    "rms": rms,
                    "sampleRate": sample_rate,
                    "stored": True,
                }
            )
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
