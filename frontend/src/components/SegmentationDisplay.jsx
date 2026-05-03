import React from 'react';
import './SegmentationDisplay.css';

function SegmentationDisplay({ result }) {
  if (!result) return null;

  return (
    <div className="segmentation-display">
      <h2>Segmentation output</h2>
      <div className="segmentation-meta">
        <span className="segmentation-method-pill">{result.method}</span>
        <p className="segmentation-desc">{result.description}</p>
        {result.parameters && (
          <dl className="segmentation-params">
            {Object.entries(result.parameters).map(([k, v]) => (
              <div key={k} className="param-row">
                <dt>{k}</dt>
                <dd>{typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(4) : String(v)}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>
      <div className="segmentation-images">
        <figure className="seg-figure">
          <figcaption>Original</figcaption>
          <img
            src={`data:image/png;base64,${result.original_image}`}
            alt="Original"
            className="seg-img"
          />
        </figure>
        <figure className="seg-figure">
          <figcaption>Segmented</figcaption>
          <img
            src={`data:image/png;base64,${result.result_image}`}
            alt="Segmented"
            className="seg-img"
          />
        </figure>
      </div>
    </div>
  );
}

export default SegmentationDisplay;
