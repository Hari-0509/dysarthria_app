import React, { useState, useRef } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [audioURL, setAudioURL] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'audio/wav') {
      setAudioFile(file);
      setAudioURL(URL.createObjectURL(file));
      setError(null);
      setResult(null);
    } else {
      setError('Please upload a WAV file');
    }
  };

  const handleAnalyze = async () => {
    if (!audioFile) {
      setError('Please select an audio file');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', audioFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(`Failed to analyze: ${err.message}`);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setAudioFile(null);
    setAudioURL(null);
    setResult(null);
    setError(null);
    fileInputRef.current.value = '';
  };

  return (
    <div className="app">
      <div className="header">
        <div className="header-content">
          <h1>Dysarthria Detection</h1>
          <p>AI-powered speech analysis using HuBERT embeddings</p>
        </div>
        <div className="status-badge">
          <span className="status-dot"></span>
          Online
        </div>
      </div>

      <div className="container">
        <div className="upload-section">
          <div className="upload-box">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".wav"
              className="file-input"
              id="audio-input"
            />
            <label htmlFor="audio-input" className="upload-label">
              <div className="upload-icon">🎤</div>
              <div className="upload-text">
                <p className="upload-title">Upload Audio File</p>
                <p className="upload-subtitle">WAV format recommended</p>
              </div>
            </label>
          </div>

          {audioURL && (
            <div className="audio-preview">
              <div className="audio-info">
                <span className="audio-name">📄 {audioFile.name}</span>
                <span className="audio-size">
                  ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
              <audio controls className="audio-player">
                <source src={audioURL} type="audio/wav" />
              </audio>
            </div>
          )}

          {error && (
            <div className="error-box">
              <span className="error-icon">⚠️</span>
              <p>{error}</p>
            </div>
          )}

          <div className="button-group">
            <button
              onClick={handleAnalyze}
              disabled={!audioFile || loading}
              className="btn btn-primary"
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                'Analyze Speech'
              )}
            </button>
            {audioFile && (
              <button onClick={handleClear} className="btn btn-secondary">
                Clear
              </button>
            )}
          </div>
        </div>

        {result && (
          <div className="result-section">
            <h2>Analysis Results</h2>
            <div className="results-grid">
              <div className={`result-card ${result.prediction.toLowerCase()}`}>
                <div className="result-label">Speech Condition</div>
                <div className="result-value">
                  {result.prediction === 'NORMAL' ? '✅' : '⚠️'}{' '}
                  {result.prediction}
                </div>
              </div>

              <div className="result-card">
                <div className="result-label">Confidence</div>
                <div className="result-value">
                  {(result.confidence * 100).toFixed(1)}%
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              {result.severity && (
                <div className={`result-card severity-${result.severity.toLowerCase()}`}>
                  <div className="result-label">Severity Level</div>
                  <div className="result-value">{result.severity}</div>
                </div>
              )}
            </div>

            <div className="clinical-info">
              <h3>📋 Clinical Information</h3>
              <p>
                {result.prediction === 'NORMAL'
                  ? 'No significant dysarthria indicators detected in the speech sample.'
                  : `Dysarthria detected with ${result.severity} severity. 
                     Recommend clinical evaluation for comprehensive assessment.`}
              </p>
            </div>
          </div>
        )}

        <div className="info-section">
          <h3>About This System</h3>
          <div className="info-grid">
            <div className="info-card">
              <div className="info-icon">🧠</div>
              <div className="info-text">
                <h4>HuBERT Model</h4>
                <p>Advanced speech representation learning for accurate feature extraction</p>
              </div>
            </div>
            <div className="info-card">
              <div className="info-icon">🤖</div>
              <div className="info-text">
                <h4>CatBoost Classifier</h4>
                <p>Gradient boosting for reliable dysarthria detection</p>
              </div>
            </div>
            <div className="info-card">
              <div className="info-icon">📊</div>
              <div className="info-text">
                <h4>Real-time Analysis</h4>
                <p>Instant predictions with detailed confidence metrics</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <footer className="footer">
        <p>Dysarthria Detection System v1.0 | Powered by FastAPI & React</p>
      </footer>
    </div>
  );
}

export default App;
