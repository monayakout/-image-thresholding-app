import axios from 'axios';

const BASE_URL = '/api';

const SEGMENT_SLUGS = {
  kmeans: 'kmeans',
  region_growing: 'region-growing',
  agglomerative: 'agglomerative',
  mean_shift: 'mean-shift',
};

export async function thresholdImage(file, params = {}) {
  const formData = new FormData();
  formData.append('image', file);
  if (params.spectral_classes) formData.append('spectral_classes', params.spectral_classes);
  if (params.local_block) formData.append('local_block', params.local_block);
  if (params.local_offset !== undefined) formData.append('local_offset', params.local_offset);

  const response = await axios.post(`${BASE_URL}/threshold/`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * @param {File} file
 * @param {'kmeans'|'region_growing'|'agglomerative'|'mean_shift'} method
 * @param {Record<string, string|number>} params — appended when defined (skips empty strings)
 */
export async function segmentImage(file, method, params = {}) {
  const slug = SEGMENT_SLUGS[method];
  if (!slug) {
    throw new Error(`Unknown segmentation method: ${method}`);
  }

  const formData = new FormData();
  formData.append('image', file);
  Object.entries(params).forEach(([key, val]) => {
    if (val === undefined || val === null || val === '') return;
    formData.append(key, val);
  });

  const response = await axios.post(`${BASE_URL}/segment/${slug}/`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function checkHealth() {
  const response = await axios.get(`${BASE_URL}/health/`);
  return response.data;
}

export async function predictFace(file) {
  const formData = new FormData();
  formData.append('image', file);

  const response = await axios.post(`${BASE_URL}/face-recognition/predict/`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}
