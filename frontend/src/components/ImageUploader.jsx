import React, { useRef } from 'react';
import './ImageUploader.css';

function ImageUploader({ onFileSelect, preview }) {
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) onFileSelect(file);
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
  };

  return (
    <div className="uploader">
      <h2>Upload Image</h2>
      <div
        className="drop-zone"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => inputRef.current.click()}
      >
        {preview ? (
          <img src={preview} alt="Selected" className="preview-img" />
        ) : (
          <div className="drop-hint">
            <span className="drop-icon">🖼</span>
            <p>Drag &amp; drop a grayscale image here</p>
            <p className="drop-sub">or click to browse</p>
            <p className="drop-sub">Supports JPG, PNG, BMP, TIFF</p>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={handleChange}
      />
    </div>
  );
}

export default ImageUploader;