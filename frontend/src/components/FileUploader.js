import React, { useState } from 'react';
import './FileUploader.css';

const FileUploader = ({ onFileUpload }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('audio/')) {
      setSelectedFile(file);
      onFileUpload(file);
    } else {
      alert('Please select a valid audio file.');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      setSelectedFile(file);
      onFileUpload(file);
    } else {
      alert('Please drop a valid audio file.');
    }
  };

  return (
    <div className="file-uploader">
      <h3>Upload Audio File</h3>
      <div 
        className={`drop-zone ${isDragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="drop-zone-content">
          <div className="upload-icon">📁</div>
          <p>Drag and drop audio file here</p>
          <p>or</p>
          <label className="file-input-label">
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileSelect}
              className="file-input"
            />
            Choose File
          </label>
        </div>
      </div>
      {selectedFile && (
        <div className="file-info">
          <h4>Selected File:</h4>
          <p>{selectedFile.name}</p>
          <p>Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
        </div>
      )}
    </div>
  );
};

export default FileUploader;