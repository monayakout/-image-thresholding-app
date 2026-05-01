import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import ThresholdControls from './components/ThresholdControls';
import ResultsDisplay from './components/ResultsDisplay';
import HistogramChart from './components/HistogramChart';
import { thresholdImage } from './api';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview]           = useState(null);
  const [loading, setLoading]           = useState(false);
  const [results, setResults]           = useState(null);
  const [error, setError]               = useState(null);
  const [params, setParams] = useState({
    spectral_classes: 3,
    local_block: 35,
    local_offset: 10,
  });

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleProcess = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await thresholdImage(selectedFile, params);
      setResults(data);
    } catch (err) {
      setError(err.message || 'Something went wrong. Is Django running on port 8000?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Image Thresholding Lab</h1>
        <p className="subtitle">
          Upload a grayscale image — apply Optimal, Otsu, Spectral, and Local thresholding
        </p>
      </header>

      <main className="app-main">
        <aside className="panel panel-left">
          <ImageUploader onFileSelect={handleFileSelect} preview={preview} />
          <ThresholdControls params={params} onChange={setParams} />
          <button
            className="btn-process"
            onClick={handleProcess}
            disabled={!selectedFile || loading}
          >
            {loading ? 'Processing…' : 'Run Thresholding'}
          </button>
          {error && <p className="error-msg">{error}</p>}
        </aside>

        <section className="panel panel-right">
          {results ? (
            <>
              <HistogramChart histogram={results.histogram} />
              <ResultsDisplay
                originalImage={results.original_image}
                results={results.results}
              />
            </>
          ) : (
            <div className="placeholder">
              <p>Upload an image and click <strong>Run Thresholding</strong> to see results.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;