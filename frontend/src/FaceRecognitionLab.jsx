import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ImageUploader from './components/ImageUploader';
import { predictFace, detectFace } from './api';

function FaceRecognitionLab() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [actionType, setActionType] = useState(null); // 'detect' or 'recognize'
  const [result, setResult] = useState(null);
  const [predictedClasses, setPredictedClasses] = useState([]);
  const [predictedImages, setPredictedImages] = useState([]);
  const [error, setError] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResult(null);
    setPredictedClasses([]);
    setPredictedImages([]);
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleProcess = async (type) => {
    if (!selectedFile) return;
    setLoading(true);
    setActionType(type);
    setError(null);
    setResult(null);
    setPredictedClasses([]);
    setPredictedImages([]);
    try {
      let data;
      if (type === 'detect') {
        data = await detectFace(selectedFile);
      } else {
        data = await predictFace(selectedFile);
        if (data.predicted_classes) {
          setPredictedClasses(data.predicted_classes);
        }
        if (data.predicted_images) {
          setPredictedImages(data.predicted_images);
        }
      }
      setResult(data.result_image);
    } catch (err) {
      setError(err.response?.data?.error || err.message || `Face ${type === 'detect' ? 'detection' : 'recognition'} failed.`);
    } finally {
      setLoading(false);
      setActionType(null);
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
        <p className="subtitle">Upload an image to detect or recognize faces.</p>
      </header>

      <main className="app-main">
        <aside className="panel panel-left">
          <ImageUploader onFileSelect={handleFileSelect} preview={preview} />

          <div style={{ display: 'flex', gap: '10px', marginTop: '15px' }}>
            <button
              type="button"
              className="btn-process"
              onClick={() => handleProcess('detect')}
              disabled={!selectedFile || loading}
              style={{ flex: 1 }}
            >
              {loading && actionType === 'detect' ? 'Processing…' : 'Face Detection'}
            </button>
            <button
              type="button"
              className="btn-process"
              onClick={() => handleProcess('recognize')}
              disabled={!selectedFile || loading}
              style={{ flex: 1 }}
            >
              {loading && actionType === 'recognize' ? 'Processing…' : 'Face Recognition'}
            </button>
          </div>

          {error && <p className="error-msg">{error}</p>}
        </aside>

        <section className="panel panel-right" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          {result !== null && (
            <div style={{ textAlign: 'center', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <h2 style={{ fontSize: '2rem', color: '#1e3a5f', marginBottom: '20px' }}>Result</h2>
              
              <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-start', justifyContent: 'center', flexWrap: 'wrap', width: '100%' }}>
                <img src={result} alt="Result" style={{ maxWidth: '100%', maxHeight: '60vh', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
                
                {predictedClasses.length > 0 && (
                  <div style={{ padding: '20px', backgroundColor: '#f8f9fa', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)', minWidth: '200px' }}>
                    <h3 style={{ marginTop: 0, color: '#1e3a5f', borderBottom: '2px solid #dee2e6', paddingBottom: '10px' }}>Predicted Classes</h3>
                    <ul style={{ listStyleType: 'none', padding: 0, margin: 0, textAlign: 'left' }}>
                      {predictedClasses.map((cls, idx) => (
                        <li key={idx} style={{ padding: '10px 0', borderBottom: '1px solid #e9ecef', fontWeight: '500', color: '#495057', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '15px' }}>
                          <div>Face {idx + 1}: <span style={{ color: '#007bff', fontWeight: 'bold' }}>{cls}</span></div>
                          {predictedImages[idx] && (
                            <img src={predictedImages[idx]} alt={`Sample of ${cls}`} style={{ width: '64px', height: '64px', borderRadius: '4px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }} title={`Sample of ${cls}`} />
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {result === null && (
            <div className="placeholder">
              <p>
                Upload an image and click <strong>Face Detection</strong> or <strong>Face Recognition</strong> to see the result.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default FaceRecognitionLab;
