# Dysarthria Detection System

AI-powered speech analysis for dysarthria detection using HuBERT embeddings and CatBoost.

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: React.js
- **ML Model**: HuBERT + CatBoost
- **Deployment**: Docker + AWS EC2
- **CI/CD**: GitHub Actions

## Local Development

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend development)

### Run Locally

```bash
docker-compose up
```

Visit:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Deployment

See AWS_SETUP.md for detailed AWS EC2 deployment instructions.

## API Endpoints

### POST /predict
Upload audio file for dysarthria detection

**Request**:
