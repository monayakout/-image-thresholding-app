import React, { useState } from 'react';
import './ResultsDisplay.css';

const METHOD_COLORS = {
  'Optimal (Isodata)': '#4f8ef7',
  "Otsu's Method":     '#22c55e',
  'Local (Adaptive)':  '#f59e0b',
};

function getColor(method) {
  for (const key of Object.keys(METHOD_COLORS)) {
    if (method.startsWith(key)) return METHOD_COLORS[key];
  }
  return '#a855f7';
}

function ResultCard({ original, result }) {
  const [showOriginal, setShowOriginal] = useState(false);
  const accentColor = getColor(result.method);

  return (
    <div className="result-card" style={{ '--accent': accentColor }}>
      <div className="result-card-header">
        <span className="method-tag" style={{ background: accentColor + '22', color: accentColor }}>
          {result.method}
        </span>
        {result.threshold_value !== null && result.threshold_value !== undefined && (
          <span className="threshold-badge">t = {result.threshold_value}</span>
        )}
        {Array.isArray(result.threshold_values) && (
          <span className="threshold-badge">t = [{result.threshold_values.join(', ')}]</span>
        )}
      </div>

      <div className="result-image-wrapper">
        <img
          src={`data:image/png;base64,${showOriginal ? original : result.result_image}`}
          alt={result.method}
          className="result-img"
        />
        <button className="toggle-btn" onClick={() => setShowOriginal(v => !v)}>
          {showOriginal ? 'Show result' : 'Show original'}
        </button>
      </div>

      <p className="result-desc">{result.description}</p>
    </div>
  );
}

function ResultsDisplay({ originalImage, results }) {
  return (
    <div className="results-display">
      <h2>Thresholding Results</h2>
      <div className="results-grid">
        <div className="result-card original-card">
          <div className="result-card-header">
            <span className="method-tag" style={{ background: '#e5e7eb', color: '#374151' }}>
              Original (grayscale)
            </span>
          </div>
          <div className="result-image-wrapper">
            <img src={`data:image/png;base64,${originalImage}`} alt="Original" className="result-img" />
          </div>
          <p className="result-desc">Input image converted to grayscale before processing.</p>
        </div>

        {results.map((r, i) => (
          <ResultCard key={i} original={originalImage} result={r} />
        ))}
      </div>
    </div>
  );
}

export default ResultsDisplay;