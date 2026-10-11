# Audio Signal Processing Platform

Real-time audio signal synthesis and analysis platform with database storage.

## Quick Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/joker9side/audio-signal-platform)

## Features

- 🎵 Real-time microphone input and analysis
- 🎛️ Signal synthesis (sine, square, triangle, saw waves)
- 📊 FFT frequency analysis
- 📈 Live waveform and spectrum visualization
- 💾 SQLite database for signal history
- 🌐 WebSocket for real-time data streaming
- 🚀 Ready for production deployment

## Local Development

### Requirements
- Python 3.11+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000/

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Synthesize Signal
```bash
curl -X POST "http://localhost:8000/api/synthesize?frequency=440&duration=1&waveform=sine"
```

### Get History
```bash
curl http://localhost:8000/api/history?limit=25
```

### WebSocket
```
ws://localhost:8000/ws/audio
```

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML5 + JavaScript (Web Audio API)
- **Database**: SQLite
- **Real-time**: WebSocket
- **Signal Processing**: NumPy, SciPy

## Usage

1. Click "Start Microphone" to begin recording
2. Set frequency and select waveform type
3. Click "Synthesize Signal" to generate tones
4. View real-time waveform and FFT spectrum
5. All data is automatically stored in database
6. View signal history at the bottom

## License

MIT
