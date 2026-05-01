import React, { useEffect, useRef } from 'react';
import './HistogramChart.css';

function HistogramChart({ histogram }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!histogram || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { values, counts } = histogram;

    const W = canvas.width;
    const H = canvas.height;
    const padL = 40, padR = 10, padT = 10, padB = 30;
    const chartW = W - padL - padR;
    const chartH = H - padT - padB;
    const maxCount = Math.max(...counts);

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, W, H);

    const barW = chartW / values.length;
    for (let i = 0; i < values.length; i++) {
      const barH = (counts[i] / maxCount) * chartH;
      const x = padL + i * barW;
      const y = padT + chartH - barH;
      ctx.fillStyle = `rgb(${values[i]},${values[i]},${values[i]})`;
      ctx.fillRect(x, y, Math.max(barW - 0.5, 1), barH);
    }

    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + chartH);
    ctx.lineTo(padL + chartW, padT + chartH);
    ctx.stroke();

    ctx.fillStyle = '#6b7280';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    [0, 64, 128, 192, 255].forEach((v) => {
      const x = padL + (v / 255) * chartW;
      ctx.fillText(v, x, H - 8);
    });

    ctx.save();
    ctx.translate(12, padT + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Count', 0, 0);
    ctx.restore();
  }, [histogram]);

  return (
    <div className="histogram-wrapper">
      <h3>Pixel Intensity Histogram</h3>
      <canvas ref={canvasRef} width={600} height={140} className="histogram-canvas" />
    </div>
  );
}

export default HistogramChart;