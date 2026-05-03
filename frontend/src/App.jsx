import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import ThresholdControls from './components/ThresholdControls';
import ResultsDisplay from './components/ResultsDisplay';
import HistogramChart from './components/HistogramChart';
import SegmentationControls from './components/SegmentationControls';
import SegmentationDisplay from './components/SegmentationDisplay';
import { thresholdImage, segmentImage } from './api';
import './App.css';

const SEG_DEFAULTS = {
  kmeans: { k: 3, max_iterations: 100 },
  region_growing: { seed_x: '', seed_y: '', threshold: 15, connectivity: 8 },
  agglomerative: { n_clusters: 4, linkage: 'ward', max_samples: 8000 },
  mean_shift: { bandwidth: '', quantile: 0.2, max_samples: 8000 },
};

function App() {
  const [activeTab, setActiveTab] = useState('threshold');

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [segmentationResults, setSegmentationResults] = useState(null);
  const [error, setError] = useState(null);

  const [params, setParams] = useState({
    spectral_classes: 3,
    local_block: 35,
    local_offset: 10,
  });

  const [segMethod, setSegMethod] = useState('kmeans');
  const [segParams, setSegParams] = useState(() => ({ ...SEG_DEFAULTS.kmeans }));

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
    setSegmentationResults(null);
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleSegMethodChange = (method) => {
    setSegMethod(method);
    setSegParams({ ...SEG_DEFAULTS[method] });
  };

  const buildSegmentationPayload = () => {
    if (segMethod === 'kmeans') {
      return { k: segParams.k, max_iterations: segParams.max_iterations };
    }
    if (segMethod === 'region_growing') {
      const p = { threshold: segParams.threshold, connectivity: segParams.connectivity };
      if (segParams.seed_x !== '' && segParams.seed_x != null) p.seed_x = segParams.seed_x;
      if (segParams.seed_y !== '' && segParams.seed_y != null) p.seed_y = segParams.seed_y;
      return p;
    }
    if (segMethod === 'agglomerative') {
      return {
        n_clusters: segParams.n_clusters,
        linkage: segParams.linkage,
        max_samples: segParams.max_samples,
      };
    }
    if (segMethod === 'mean_shift') {
      const p = {
        quantile: segParams.quantile,
        max_samples: segParams.max_samples,
      };
      if (segParams.bandwidth !== '' && segParams.bandwidth != null) {
        p.bandwidth = segParams.bandwidth;
      }
      return p;
    }
    return {};
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

  const handleSegmentation = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setSegmentationResults(null);
    try {
      const data = await segmentImage(selectedFile, segMethod, buildSegmentationPayload());
      setSegmentationResults(data);
    } catch (err) {
      const msg = err.response?.data?.error || err.message || 'Segmentation failed.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Image Processing Lab</h1>
        <p className="subtitle">
          Part 1: thresholding · Part 2: unsupervised segmentation
        </p>
        <nav className="app-tabs" aria-label="Lab sections">
          <button
            type="button"
            className={`app-tab ${activeTab === 'threshold' ? 'app-tab-active' : ''}`}
            onClick={() => setActiveTab('threshold')}
          >
            Part 1: Thresholding
          </button>
          <button
            type="button"
            className={`app-tab ${activeTab === 'segmentation' ? 'app-tab-active' : ''}`}
            onClick={() => setActiveTab('segmentation')}
          >
            Part 2: Segmentation
          </button>
        </nav>
      </header>

      <main className="app-main">
        <aside className="panel panel-left">
          <ImageUploader onFileSelect={handleFileSelect} preview={preview} />

          {activeTab === 'threshold' && (
            <>
              <ThresholdControls params={params} onChange={setParams} />
              <button
                type="button"
                className="btn-process"
                onClick={handleProcess}
                disabled={!selectedFile || loading}
              >
                {loading ? 'Processing…' : 'Run Thresholding'}
              </button>
            </>
          )}

          {activeTab === 'segmentation' && (
            <>
              <SegmentationControls
                method={segMethod}
                params={segParams}
                onMethodChange={handleSegMethodChange}
                onParamsChange={setSegParams}
              />
              <button
                type="button"
                className="btn-process"
                onClick={handleSegmentation}
                disabled={!selectedFile || loading}
              >
                {loading ? 'Processing…' : 'Run Segmentation'}
              </button>
            </>
          )}

          {error && <p className="error-msg">{error}</p>}
        </aside>

        <section className="panel panel-right">
          {activeTab === 'threshold' && results && (
            <>
              <HistogramChart histogram={results.histogram} />
              <ResultsDisplay
                originalImage={results.original_image}
                results={results.results}
              />
            </>
          )}

          {activeTab === 'segmentation' && segmentationResults && (
            <SegmentationDisplay result={segmentationResults} />
          )}

          {activeTab === 'threshold' && !results && (
            <div className="placeholder">
              <p>
                Upload an image and click <strong>Run Thresholding</strong>.
              </p>
            </div>
          )}

          {activeTab === 'segmentation' && !segmentationResults && (
            <div className="placeholder">
              <p>
                Upload an image, choose a method, then click <strong>Run Segmentation</strong>.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
