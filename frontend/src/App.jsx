import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home';
import ImageProcessingLab from './ImageProcessingLab';
import FaceRecognitionLab from './FaceRecognitionLab';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/image-processing" element={<ImageProcessingLab />} />
        <Route path="/face-recognition" element={<FaceRecognitionLab />} />
      </Routes>
    </Router>
  );
}

export default App;
