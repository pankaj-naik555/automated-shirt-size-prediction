// Frontend Application Logic
// Handles UI interactions, API calls, and WebSocket communication

// API Configuration
const API_BASE_URL = window.location.origin;
const socket = io(API_BASE_URL);

// Global State
let currentPersonId = null;
let cameraActive = false;
let currentMeasurements = null;
let frameRequestInterval = null;

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Application initialized');
    
    // Initialize components
    loadPersonList();
    loadStatistics();
    setupEventListeners();
    setupWebSocket();
    
    // Check camera source selection
    document.getElementById('cameraSource').addEventListener('change', function() {
        const ipDiv = document.getElementById('ipCameraDiv');
        if (this.value === 'ip') {
            ipDiv.classList.remove('d-none');
        } else {
            ipDiv.classList.add('d-none');
        }
    });
});

// ==================== Event Listeners ====================

function setupEventListeners() {
    // Person Form
    document.getElementById('personForm').addEventListener('submit', handlePersonRegistration);
    
    // Person Selection
    document.getElementById('selectPerson').addEventListener('change', handlePersonSelection);
    
    // Camera Controls
    document.getElementById('startCameraBtn').addEventListener('click', startCamera);
    document.getElementById('stopCameraBtn').addEventListener('click', stopCamera);
    
    // Find Size Button
    document.getElementById('findSizeBtn').addEventListener('click', captureAndPredict);
    
    // Save Result
    document.getElementById('saveResultBtn').addEventListener('click', saveResult);
    
    // Export Result
    document.getElementById('exportResultBtn').addEventListener('click', exportResult);
    
    // Refresh Statistics
    document.getElementById('refreshStatsBtn').addEventListener('click', loadStatistics);
}

// ==================== WebSocket Setup ====================

function setupWebSocket() {
    socket.on('connect', function() {
        console.log('WebSocket connected');
        showNotification('Connected to server', 'success');
    });
    
    socket.on('disconnect', function() {
        console.log('WebSocket disconnected');
        showNotification('Disconnected from server', 'warning');
    });
    
    socket.on('camera_started', function(data) {
        console.log('Camera started:', data);
        cameraActive = true;
        updateCameraStatus('connected', 'Connected');
        document.getElementById('cameraPlaceholder').style.display = 'none';
        document.getElementById('cameraFeed').style.display = 'block';
        document.getElementById('startCameraBtn').classList.add('d-none');
        document.getElementById('stopCameraBtn').classList.remove('d-none');
        
        // Start requesting frames
        startFrameRequest();
    });
    
    socket.on('camera_stopped', function(data) {
        console.log('Camera stopped:', data);
        cameraActive = false;
        updateCameraStatus('secondary', 'Not Connected');
        document.getElementById('cameraPlaceholder').style.display = 'flex';
        document.getElementById('cameraFeed').style.display = 'none';
        document.getElementById('startCameraBtn').classList.remove('d-none');
        document.getElementById('stopCameraBtn').classList.add('d-none');
        document.getElementById('findSizeBtn').disabled = true;
        
        // Stop requesting frames
        stopFrameRequest();
    });
    
    socket.on('frame_data', function(data) {
        // Update camera feed
        const img = document.getElementById('cameraFeed');
        img.src = 'data:image/jpeg;base64,' + data.frame;
        
        // Update measurements
        if (data.measurements) {
            currentMeasurements = data.measurements;
            updateMeasurementsDisplay(data.measurements);
            updatePoseStatus(true);
            document.getElementById('findSizeBtn').disabled = false;
        } else {
            updatePoseStatus(false);
            document.getElementById('findSizeBtn').disabled = true;
        }
    });
    
    socket.on('prediction_result', function(data) {
        console.log('Prediction result:', data);
        if (data.success) {
            displayPrediction(data.prediction);
            showNotification('Size predicted successfully!', 'success');
            loadMeasurementHistory(currentPersonId);
        }
    });
    
    socket.on('camera_error', function(data) {
        console.error('Camera error:', data);
        showNotification('Camera error: ' + data.error, 'danger');
        updateCameraStatus('error', 'Error');
    });
    
    socket.on('frame_error', function(data) {
        console.error('Frame error:', data);
    });
    
    socket.on('prediction_error', function(data) {
        console.error('Prediction error:', data);
        showNotification('Prediction error: ' + data.error, 'danger');
    });
}

// ==================== Person Management ====================

async function handlePersonRegistration(e) {
    e.preventDefault();
    
    const personData = {
        name: document.getElementById('personName').value,
        email: document.getElementById('personEmail').value || null,
        phone: document.getElementById('personPhone').value || null,
        actual_shirt_size: document.getElementById('actualSize').value || null
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/person/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(personData)
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showNotification('Person registered successfully!', 'success');
            currentPersonId = data.person.id;
            updateCurrentPersonDisplay(data.person);
            document.getElementById('personForm').reset();
            loadPersonList();
        } else {
            showNotification('Error: ' + data.error, 'danger');
        }
    } catch (error) {
        console.error('Error registering person:', error);
        showNotification('Failed to register person', 'danger');
    }
}

async function loadPersonList() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/person/all`);
        const data = await response.json();
        
        if (data.success) {
            const select = document.getElementById('selectPerson');
            select.innerHTML = '<option value="">Select a person...</option>';
            
            data.persons.forEach(person => {
                const option = document.createElement('option');
                option.value = person.id;
                option.textContent = `${person.name} (${person.measurement_count} measurements)`;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading person list:', error);
    }
}

function handlePersonSelection(e) {
    const personId = parseInt(e.target.value);
    if (personId) {
        loadPersonDetails(personId);
    } else {
        currentPersonId = null;
        document.getElementById('currentPerson').classList.add('d-none');
    }
}

async function loadPersonDetails(personId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/person/${personId}`);
        const data = await response.json();
        
        if (data.success) {
            currentPersonId = personId;
            updateCurrentPersonDisplay(data.person);
            loadMeasurementHistory(personId);
        }
    } catch (error) {
        console.error('Error loading person details:', error);
    }
}

function updateCurrentPersonDisplay(person) {
    const currentPersonDiv = document.getElementById('currentPerson');
    const currentPersonName = document.getElementById('currentPersonName');
    
    currentPersonName.textContent = person.name;
    currentPersonDiv.classList.remove('d-none');
}

// ==================== Camera Management ====================

function startCamera() {
    if (!currentPersonId) {
        showNotification('Please select or register a person first', 'warning');
        return;
    }
    
    let cameraSource = document.getElementById('cameraSource').value;
    
    if (cameraSource === 'ip') {
        cameraSource = document.getElementById('ipCameraUrl').value;
        if (!cameraSource) {
            showNotification('Please enter IP camera URL', 'warning');
            return;
        }
    }
    
    const heightInput = document.getElementById('heightInput').value;
    
    socket.emit('start_camera', {
        camera_source: cameraSource,
        person_id: currentPersonId,
        estimated_height_cm: heightInput ? parseFloat(heightInput) : null
    });
    
    updateCameraStatus('warning', 'Connecting...');
}

function stopCamera() {
    socket.emit('stop_camera');
}

function startFrameRequest() {
    // Request frames at 15 FPS
    frameRequestInterval = setInterval(() => {
        if (cameraActive) {
            const heightInput = document.getElementById('heightInput').value;
            socket.emit('request_frame', {
                estimated_height_cm: heightInput ? parseFloat(heightInput) : null
            });
        }
    }, 66); // ~15 FPS
}

function stopFrameRequest() {
    if (frameRequestInterval) {
        clearInterval(frameRequestInterval);
        frameRequestInterval = null;
    }
}

function updateCameraStatus(type, text) {
    const statusBadge = document.getElementById('cameraStatus');
    statusBadge.className = 'badge bg-' + type;
    if (type === 'success') {
        statusBadge.className = 'badge connected';
    } else if (type === 'danger') {
        statusBadge.className = 'badge error';
    }
    statusBadge.textContent = text;
}

// ==================== Measurements Display ====================

function updateMeasurementsDisplay(measurements) {
    document.getElementById('heightValue').textContent = measurements.height_cm.toFixed(1) + ' cm';
    document.getElementById('chestValue').textContent = measurements.chest_cm.toFixed(1) + ' cm';
    document.getElementById('waistValue').textContent = measurements.waist_cm.toFixed(1) + ' cm';
    document.getElementById('shoulderValue').textContent = measurements.shoulder_width_cm.toFixed(1) + ' cm';
    document.getElementById('armValue').textContent = measurements.arm_length_cm.toFixed(1) + ' cm';
    document.getElementById('weightValue').textContent = measurements.weight_kg.toFixed(1) + ' kg';
}

function updatePoseStatus(detected) {
    const statusBadge = document.getElementById('poseStatus');
    if (detected) {
        statusBadge.className = 'badge pose-detected';
        statusBadge.innerHTML = '<i class="bi bi-check-circle"></i> Pose Detected';
    } else {
        statusBadge.className = 'badge bg-secondary';
        statusBadge.innerHTML = '<i class="bi bi-circle"></i> No Pose Detected';
    }
}

// ==================== Prediction ====================

function captureAndPredict() {
    if (!currentPersonId) {
        showNotification('Please select a person first', 'warning');
        return;
    }
    
    if (!currentMeasurements) {
        showNotification('No measurements available', 'warning');
        return;
    }
    
    socket.emit('capture_and_predict', {
        person_id: currentPersonId
    });
    
    showNotification('Processing prediction...', 'info');
}

function displayPrediction(prediction) {
    // Show prediction card
    const card = document.getElementById('predictionCard');
    card.classList.remove('d-none');
    card.classList.add('fade-in');
    
    // Update predicted size
    document.getElementById('predictedSize').textContent = prediction.predicted_size;
    
    // Update confidence bar
    const confidence = prediction.confidence * 100;
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    
    confidenceBar.style.width = confidence + '%';
    confidenceText.textContent = confidence.toFixed(1) + '%';
    
    // Set color based on confidence
    if (confidence >= 80) {
        confidenceBar.className = 'progress-bar bg-success';
    } else if (confidence >= 60) {
        confidenceBar.className = 'progress-bar bg-warning';
    } else {
        confidenceBar.className = 'progress-bar bg-danger';
    }
    
    // Update size distribution
    const distributionDiv = document.getElementById('sizeDistribution');
    distributionDiv.innerHTML = '';
    
    const sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL'];
    sizes.forEach(size => {
        const prob = (prediction.all_probabilities[size] || 0) * 100;
        
        const barDiv = document.createElement('div');
        barDiv.className = 'size-bar mb-2';
        barDiv.innerHTML = `
            <div class="d-flex align-items-center">
                <span class="size-bar-label">${size}</span>
                <div class="flex-grow-1 mx-2">
                    <div class="progress" style="height: 25px;">
                        <div class="progress-bar" style="width: ${prob}%">
                            ${prob.toFixed(1)}%
                        </div>
                    </div>
                </div>
            </div>
        `;
        distributionDiv.appendChild(barDiv);
    });
    
    // Scroll to prediction
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function saveResult() {
    showNotification('Result saved to database!', 'success');
}

function exportResult() {
    showNotification('Export feature coming soon!', 'info');
}

// ==================== Measurement History ====================

async function loadMeasurementHistory(personId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/measurement/history/${personId}`);
        const data = await response.json();
        
        if (data.success) {
            const tbody = document.getElementById('historyTableBody');
            tbody.innerHTML = '';
            
            if (data.measurements.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="8" class="text-center text-muted">
                            No measurements yet
                        </td>
                    </tr>
                `;
                return;
            }
            
            data.measurements.forEach(m => {
                const row = document.createElement('tr');
                const date = new Date(m.measurement_date).toLocaleDateString();
                const confidence = (m.confidence * 100).toFixed(1);
                
                let confidenceClass = 'high';
                if (m.confidence < 0.8) confidenceClass = 'medium';
                if (m.confidence < 0.6) confidenceClass = 'low';
                
                row.innerHTML = `
                    <td>${date}</td>
                    <td>${currentPersonId}</td>
                    <td><span class="size-badge ${m.predicted_size.toLowerCase()}">${m.predicted_size}</span></td>
                    <td><span class="confidence-badge ${confidenceClass}">${confidence}%</span></td>
                    <td>${m.height_cm.toFixed(1)} cm</td>
                    <td>${m.chest_cm.toFixed(1)} cm</td>
                    <td>${m.waist_cm.toFixed(1)} cm</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary" onclick="viewMeasurementDetails(${m.id})">
                            <i class="bi bi-eye"></i>
                        </button>
                    </td>
                `;
                tbody.appendChild(row);
            });
        }
    } catch (error) {
        console.error('Error loading measurement history:', error);
    }
}

function viewMeasurementDetails(measurementId) {
    showNotification('Viewing details for measurement #' + measurementId, 'info');
    // TODO: Implement modal or detailed view
}

// ==================== Statistics ====================

async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/statistics`);
        const data = await response.json();
        
        if (data.success) {
            const stats = data.statistics;
            
            document.getElementById('totalPersons').textContent = stats.total_persons;
            document.getElementById('totalMeasurements').textContent = stats.total_measurements;
            document.getElementById('avgConfidence').textContent = 
                (stats.avg_confidence * 100).toFixed(1) + '%';
        }
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// ==================== Utility Functions ====================

function showNotification(message, type = 'info') {
    const toast = document.getElementById('notificationToast');
    const toastTitle = document.getElementById('toastTitle');
    const toastMessage = document.getElementById('toastMessage');
    
    const titles = {
        'success': 'Success',
        'danger': 'Error',
        'warning': 'Warning',
        'info': 'Information'
    };
    
    toastTitle.textContent = titles[type] || 'Notification';
    toastMessage.textContent = message;
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}

// ==================== Debug Functions ====================

window.debugState = function() {
    console.log('Current State:', {
        personId: currentPersonId,
        cameraActive: cameraActive,
        measurements: currentMeasurements
    });
};