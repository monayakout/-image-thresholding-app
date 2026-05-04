import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ImageUploader from './components/ImageUploader';
import { predictFace } from './api';

function FaceRecognitionLab() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleProcess = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictFace(selectedFile);
      setResult(data.result_image);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Face recognition failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <button 
            type="button" 
            className="app-tab" 
            onClick={() => navigate('/')}
          >
            ← Back to Home
          </button>
          <h1>Face Recognition Module</h1>
        </div>
        <p className="subtitle">Upload an image to recognize the face subject.</p>
      </header>

      <main className="app-main">
        <aside className="panel panel-left">
          <ImageUploader onFileSelect={handleFileSelect} preview={preview} />

          <button
            type="button"
            className="btn-process"
            onClick={handleProcess}
            disabled={!selectedFile || loading}
          >
            {loading ? 'Processing…' : 'Run Face Recognition'}
          </button>

          {error && <p className="error-msg">{error}</p>}
        </aside>

        <section className="panel panel-right" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          {result !== null && (
            <div style={{ textAlign: 'center' }}>
              <h2 style={{ fontSize: '2rem', color: '#1e3a5f', marginBottom: '10px' }}>Recognition Result</h2>
              <img src={result} alt="Recognition Result" style={{ maxWidth: '100%', maxHeight: '60vh', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
            </div>
          )}

          {result === null && (
            <div className="placeholder">
              <p>
                Upload an image and click <strong>Run Face Recognition</strong> to see the result.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default FaceRecognitionLab;
