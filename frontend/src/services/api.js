import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeAudio = async (audioData, isFile = false) => {
  try {
    let response;
    
    if (isFile) {
      // Handle file upload
      const formData = new FormData();
      formData.append('file', audioData);
      
      response = await api.post('/analyze-file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    } else {
      // Handle recorded audio (base64)
      const formData = new URLSearchParams();
      formData.append('audio_data', audioData);
      
      response = await api.post('/analyze-recording', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
    }
    
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get('/model-info');
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const trainModels = async () => {
  try {
    const response = await api.post('/train-models');
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export default api;