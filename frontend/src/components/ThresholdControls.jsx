import React from 'react';
import './ThresholdControls.css';

function ThresholdControls({ params, onChange }) {
  const set = (key, value) => onChange({ ...params, [key]: value });

  return (
    <div className="controls">
      <h2>Parameters</h2>

      <div className="control-group">
        <label>
          Spectral classes
          <span className="control-value">{params.spectral_classes}</span>
        </label>
        <input type="range" min={3} max={8} value={params.spectral_classes}
          onChange={(e) => set('spectral_classes', parseInt(e.target.value))} />
        <small>Number of regions for Multi-Otsu (min 3)</small>
      </div>

      <div className="control-group">
        <label>
          Local block size
          <span className="control-value">{params.local_block}</span>
        </label>
        <input type="range" min={11} max={101} step={2} value={params.local_block}
          onChange={(e) => set('local_block', parseInt(e.target.value))} />
        <small>Neighbourhood size for adaptive thresholding</small>
      </div>

      <div className="control-group">
        <label>
          Local offset
          <span className="control-value">{params.local_offset}</span>
        </label>
        <input type="range" min={-20} max={30} step={1} value={params.local_offset}
          onChange={(e) => set('local_offset', parseFloat(e.target.value))} />
        <small>Subtracted from local mean</small>
      </div>
    </div>
  );
}

export default ThresholdControls;