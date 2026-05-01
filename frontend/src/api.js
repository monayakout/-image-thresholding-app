import axios from 'axios';

const BASE_URL = '/api';

export async function thresholdImage(file, params = {}) {
  const formData = new FormData();
  formData.append('image', file);
  if (params.spectral_classes) formData.append('spectral_classes', params.spectral_classes);
  if (params.local_block)      formData.append('local_block', params.local_block);
  if (params.local_offset !== undefined) formData.append('local_offset', params.local_offset);

  const response = await axios.post(`${BASE_URL}/threshold/`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function checkHealth() {
  const response = await axios.get(`${BASE_URL}/health/`);
  return response.data;
}