import React from 'react';
import { useNavigate } from 'react-router-dom';

function Home() {
  const navigate = useNavigate();

  return (
    <div className="app">
      <header className="app-header">
        <h1>Computer Vision Modules</h1>
        <p className="subtitle">Select a module to begin</p>
      </header>

      <main className="app-main" style={{ justifyContent: 'center', alignItems: 'center' }}>
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '100%', maxWidth: '500px', padding: '40px' }}>
          <button 
            type="button" 
            className="btn-process" 
            style={{ padding: '20px', fontSize: '1.2rem' }}
            onClick={() => navigate('/face-recognition')}
          >
            Face Recognition Module
          </button>
          
          <button 
            type="button" 
            className="btn-process" 
            style={{ padding: '20px', fontSize: '1.2rem', background: '#475569' }}
            onClick={() => navigate('/image-processing')}
          >
            Image Processing Module
          </button>
        </div>
      </main>
    </div>
  );
}

export default Home;
