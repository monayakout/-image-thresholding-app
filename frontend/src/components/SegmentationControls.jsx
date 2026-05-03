import React from 'react';
import './SegmentationControls.css';

const METHODS = [
  { id: 'kmeans', label: 'K-means' },
  { id: 'region_growing', label: 'Region Growing' },
  { id: 'agglomerative', label: 'Agglomerative' },
  { id: 'mean_shift', label: 'Mean Shift' },
];

function SegmentationControls({ method, params, onMethodChange, onParamsChange }) {
  const setField = (key, value) => {
    onParamsChange({ ...params, [key]: value });
  };

  return (
    <div className="segmentation-controls">
      <h3 className="panel-section-title">Segmentation method</h3>
      <label className="field-label" htmlFor="seg-method">Algorithm</label>
      <select
        id="seg-method"
        className="field-select"
        value={method}
        onChange={(e) => onMethodChange(e.target.value)}
      >
        {METHODS.map((m) => (
          <option key={m.id} value={m.id}>{m.label}</option>
        ))}
      </select>

      {method === 'kmeans' && (
        <>
          <label className="field-label" htmlFor="seg-k">K (clusters)</label>
          <input
            id="seg-k"
            type="number"
            min={2}
            max={32}
            className="field-input"
            value={params.k ?? 3}
            onChange={(e) => setField('k', Number(e.target.value))}
          />
          <label className="field-label" htmlFor="seg-max-iter">Max iterations</label>
          <input
            id="seg-max-iter"
            type="number"
            min={1}
            max={500}
            className="field-input"
            value={params.max_iterations ?? 100}
            onChange={(e) => setField('max_iterations', Number(e.target.value))}
          />
        </>
      )}

      {method === 'region_growing' && (
        <>
          <label className="field-label" htmlFor="seg-seed-x">Seed X (column)</label>
          <input
            id="seg-seed-x"
            type="number"
            min={0}
            className="field-input"
            placeholder="default: center"
            value={params.seed_x ?? ''}
            onChange={(e) => setField('seed_x', e.target.value === '' ? '' : Number(e.target.value))}
          />
          <label className="field-label" htmlFor="seg-seed-y">Seed Y (row)</label>
          <input
            id="seg-seed-y"
            type="number"
            min={0}
            className="field-input"
            placeholder="default: center"
            value={params.seed_y ?? ''}
            onChange={(e) => setField('seed_y', e.target.value === '' ? '' : Number(e.target.value))}
          />
          <label className="field-label" htmlFor="seg-thresh-rg">Similarity threshold</label>
          <input
            id="seg-thresh-rg"
            type="number"
            min={0}
            step={0.5}
            className="field-input"
            value={params.threshold ?? 15}
            onChange={(e) => setField('threshold', Number(e.target.value))}
          />
          <small className="field-hint">
            Grayscale: intensity difference vs seed. Color: max RGB Euclidean distance (try ~25–80 for photos).
          </small>
          <label className="field-label" htmlFor="seg-conn">Connectivity</label>
          <select
            id="seg-conn"
            className="field-select"
            value={params.connectivity ?? 8}
            onChange={(e) => setField('connectivity', Number(e.target.value))}
          >
            <option value={4}>4-neighbors</option>
            <option value={8}>8-neighbors</option>
          </select>
        </>
      )}

      {method === 'agglomerative' && (
        <>
          <label className="field-label" htmlFor="seg-n-clust">N clusters</label>
          <input
            id="seg-n-clust"
            type="number"
            min={2}
            max={64}
            className="field-input"
            value={params.n_clusters ?? 4}
            onChange={(e) => setField('n_clusters', Number(e.target.value))}
          />
          <label className="field-label" htmlFor="seg-linkage">Linkage</label>
          <select
            id="seg-linkage"
            className="field-select"
            value={params.linkage ?? 'ward'}
            onChange={(e) => setField('linkage', e.target.value)}
          >
            <option value="ward">Ward (multivariate)</option>
            <option value="average">Average</option>
            <option value="complete">Complete</option>
            <option value="single">Single</option>
          </select>
          <label className="field-label" htmlFor="seg-max-samples-agg">Max sample pixels</label>
          <input
            id="seg-max-samples-agg"
            type="number"
            min={1000}
            max={50000}
            step={500}
            className="field-input"
            value={params.max_samples ?? 8000}
            onChange={(e) => setField('max_samples', Number(e.target.value))}
          />
        </>
      )}

      {method === 'mean_shift' && (
        <>
          <label className="field-label" htmlFor="seg-bandwidth">Bandwidth (empty = auto)</label>
          <input
            id="seg-bandwidth"
            type="text"
            className="field-input"
            placeholder="auto"
            value={params.bandwidth ?? ''}
            onChange={(e) => setField('bandwidth', e.target.value)}
          />
          <label className="field-label" htmlFor="seg-quantile">Bandwidth quantile (auto)</label>
          <input
            id="seg-quantile"
            type="number"
            min={0.05}
            max={0.9}
            step={0.05}
            className="field-input"
            value={params.quantile ?? 0.2}
            onChange={(e) => setField('quantile', Number(e.target.value))}
          />
          <label className="field-label" htmlFor="seg-max-samples-ms">Max sample pixels</label>
          <input
            id="seg-max-samples-ms"
            type="number"
            min={1000}
            max={50000}
            step={500}
            className="field-input"
            value={params.max_samples ?? 8000}
            onChange={(e) => setField('max_samples', Number(e.target.value))}
          />
        </>
      )}
    </div>
  );
}

export default SegmentationControls;
