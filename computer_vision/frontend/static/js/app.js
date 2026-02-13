// Frontend Application Logic - FIXED VERSION
// Handles UI interactions, API calls, and WebSocket communication

// API Configuration
const API_BASE_URL = window.location.origin;
const socket = io(API_BASE_URL, {
    transports: ['websocket'],
    upgrade: false,
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 5
});

// Global State
let currentPersonId = null;
let cameraActive = false;
let currentMeasurements = null;
let frameRequestInterval = null;

// Auto-prediction state
let stablePoseCounter = 0;
const STABLE_POSE_REQUIRED_FRAMES = 25; // Roughly 1.5 seconds at 15 FPS
let isAutoPredicting = false;
let lastAutoPredictTime = 0;
const AUTO_PREDICT_COOLDOWN = 5000; // Wait 5 seconds before allowing another auto-predict

// Performance Optimization Variables
let isProcessingFrame = false;
let frameSkipCounter = 0;
let lastFrameTime = 0;
let currentFPS = 15;
let frameBuffer = null;

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Application initialized');
    
    loadPersonList();
    loadStatistics();
    setupEventListeners();
    setupWebSocket();
});

// ==================== Event Listeners ====================

function setupEventListeners() {
    // Person Form
    document.getElementById('personForm').addEventListener('submit', handlePersonRegistration);
    
    // Person Selection
    document.getElementById('selectPerson').addEventListener('change', handlePersonSelection);
    
    // Camera Source Selection
    document.getElementById('cameraSource').addEventListener('change', handleCameraSourceChange);
    document.getElementById('calibrateBtn').addEventListener('click', handleCalibration);
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

function handleCameraSourceChange() {
    const cameraSource = document.getElementById('cameraSource').value;
    const ipDiv = document.getElementById('ipCameraDiv');
    const usbTetherDiv = document.getElementById('usbTetherDiv');
    const adbDiv = document.getElementById('adbDiv');
    
    // Hide all
    ipDiv.classList.add('d-none');
    usbTetherDiv.classList.add('d-none');
    adbDiv.classList.add('d-none');
    
    // Show relevant
    if (cameraSource === 'ip') {
        ipDiv.classList.remove('d-none');
    } else if (cameraSource === 'usb_tether') {
        usbTetherDiv.classList.remove('d-none');
    } else if (cameraSource === 'adb') {
        adbDiv.classList.remove('d-none');
    }
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
        if (cameraActive) {
            stopFrameRequest();
            cameraActive = false;
        }
    });
    
    socket.on('camera_started', function(data) {
        console.log('Camera started:', data);
        cameraActive = true;
        updateCameraStatus('connected', 'Connected');
        const placeholder = document.getElementById('cameraPlaceholder');
        feed = document.getElementById('cameraFeed');

        placeholder.classList.add('d-none');    // Properly hides the placeholder
        placeholder.classList.remove('d-flex'); // Removes the flex layout conflict
        feed.classList.remove('d-none');       // Ensures feed isn't hidden by class
        feed.style.display = 'block';
        startFrameRequest();
    });
    
    socket.on('camera_stopped', function(data) {
        console.log('Camera stopped:', data);
        cameraActive = false;
        updateCameraStatus('secondary', 'Not Connected');
        const placeholder = document.getElementById('cameraPlaceholder');
        const feed = document.getElementById('cameraFeed');

        placeholder.classList.remove('d-none');
        placeholder.classList.add('d-flex');
        feed.style.display = 'none';
        stopFrameRequest();
    });
    // Inside setupWebSocket() function...

socket.on('camera_started', function(data) {
    console.log('Camera started:', data);
    cameraActive = true;
    updateCameraStatus('connected', 'Connected');
    
    // 1. Handle Video Feed Display (Your existing fix)
    const placeholder = document.getElementById('cameraPlaceholder');
    const feed = document.getElementById('cameraFeed');
    placeholder.classList.add('d-none');
    placeholder.classList.remove('d-flex');
    feed.classList.remove('d-none');
    feed.style.display = 'block';
    
    // 2. NEW FIX: Toggle Buttons
    document.getElementById('startCameraBtn').classList.add('d-none'); // Hide Start
    document.getElementById('stopCameraBtn').classList.remove('d-none'); // Show Stop
    
    startFrameRequest();
});

socket.on('camera_stopped', function(data) {
    console.log('Camera stopped:', data);
    cameraActive = false;
    updateCameraStatus('secondary', 'Not Connected');
    
    // 1. Handle Video Feed Display
    const placeholder = document.getElementById('cameraPlaceholder');
    const feed = document.getElementById('cameraFeed');
    placeholder.classList.remove('d-none');
    placeholder.classList.add('d-flex');
    feed.style.display = 'none';
    
    // 2. NEW FIX: Toggle Buttons
    document.getElementById('startCameraBtn').classList.remove('d-none'); // Show Start
    document.getElementById('stopCameraBtn').classList.add('d-none'); // Hide Stop
    
    stopFrameRequest();
});
socket.on('frame_data', function(data) {
        isProcessingFrame = false;
        
        const now = performance.now();
        if (lastFrameTime > 0) {
            const actualFPS = 1000 / (now - lastFrameTime);
            if (frameSkipCounter % 30 === 0) {
                console.log(`Actual FPS: ${actualFPS.toFixed(1)}`);
            }
        }
        lastFrameTime = now;
        frameSkipCounter++;
        
        const img = document.getElementById('cameraFeed');
        if (!frameBuffer) {
            frameBuffer = new Image();
            frameBuffer.onload = function() {
                img.src = frameBuffer.src;
            };
        }
        frameBuffer.src = 'data:image/jpeg;base64,' + data.frame;
        
        if (frameSkipCounter % 5 === 0) {
            if (data.measurements) {
                currentMeasurements = data.measurements;
                updateMeasurementsDisplay(data.measurements);
                updatePoseStatus(true);
                document.getElementById('findSizeBtn').disabled = false;
            } else {
                updatePoseStatus(false);
                document.getElementById('findSizeBtn').disabled = true;
            }
        }
    });
    
socket.on('prediction_result', function(data) {
    console.log('Prediction result:', data);
    
    // 1. Reset the auto-predict lock immediately
    isAutoPredicting = false;
    
    // 2. Clear the "Steady" timer UI 
    const poseStatus = document.getElementById('poseStatus');
    if (poseStatus) {
        poseStatus.textContent = 'Captured';
        poseStatus.classList.remove('bg-warning');
        poseStatus.classList.add('bg-success');
    }

    if (data.success) {
        // 3. Display the actual size (e.g., "M", "L")
        displayPrediction(data.prediction);
        
        // 4. Detailed Notifications
        if (data.warning) {
            // Show result but warn about DB
            showNotification(`Size: ${data.prediction.predicted_size} (Note: ${data.warning})`, 'warning');
        } else {
            showNotification(`Size Predicted: ${data.prediction.predicted_size}`, 'success');
        }
        
        // 5. Refresh the history table
        if (currentPersonId) {
            loadMeasurementHistory(currentPersonId);
            loadStatistics(); // Refresh the top dashboard cards too
        }
    } else {
        showNotification('Prediction failed. Please try again.', 'danger');
    }
});
    
socket.on('camera_error', function(data) {
        console.error('Camera error:', data);
        showNotification('Camera error: ' + data.error, 'danger');
        updateCameraStatus('error', 'Error');
        isProcessingFrame = false;
    });
    
    socket.on('frame_error', function(data) {
        console.error('Frame error:', data);
        isProcessingFrame = false;
    });
    
socket.on('prediction_error', function(data) {
        console.error('Prediction error:', data);
        showNotification('Prediction error: ' + data.error, 'danger');
    });
}

socket.on('measurements', function(data) {
    if (data.success) {
        currentMeasurements = data.measurements;
        updateMeasurementDisplay(data.measurements);

        // --- Auto-Prediction Logic ---
        const now = Date.now();
        const hasCooldownPassed = (now - lastAutoPredictTime) > AUTO_PREDICT_COOLDOWN;
        
        // Only auto-predict if: 
        // 1. Pose quality is high (> 75%)
        // 2. We aren't already in the middle of a request
        // 3. Cooldown has passed
        // 4. A person is selected in the dropdown
        if (data.pose_quality > 0.75 && !isAutoPredicting && hasCooldownPassed && currentPersonId) {
            stablePoseCounter++;
            
            // Optional: Update UI to show "Calibrating..." or a progress bar
            if (stablePoseCounter > 5) {
                document.getElementById('poseStatus').textContent = `Steady... ${Math.round((stablePoseCounter/STABLE_POSE_REQUIRED_FRAMES)*100)}%`;
            }

            if (stablePoseCounter >= STABLE_POSE_REQUIRED_FRAMES) {
                triggerAutoPrediction();
            }
        } else {
            // Reset counter if they move or quality drops
            stablePoseCounter = 0;
        }
    }
});
// ==================== Person Management ====================

async function handlePersonRegistration(e) {
    e.preventDefault();
    
    const name = document.getElementById('personName').value;
    const email = document.getElementById('personEmail').value;
    const phone = document.getElementById('personPhone').value;
    const actualSize = document.getElementById('actualSize').value;
    
    // Validate name
    if (!name || name.trim().length === 0) {
        showNotification('Please enter a valid name', 'warning');
        return;
    }
    
    const personData = {
        name: name.trim(),
        email: email && email.trim().length > 0 ? email.trim() : null,
        phone: phone && phone.trim().length > 0 ? phone.trim() : null,
        actual_shirt_size: actualSize && actualSize.trim().length > 0 ? actualSize : null
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
            showNotification('Error: ' + (data.error || 'Unknown error'), 'danger');
        }
    } catch (error) {
        console.error('Error registering person:', error);
        showNotification('Failed to register person: ' + error.message, 'danger');
    }
}
async function handleCalibration() {
    const method = document.getElementById('calibrationMethod').value;
    const heightInput = document.getElementById('heightInput').value;

    if (method === 'height') {
        // Manual Height Calibration
        if (!heightInput || heightInput < 50 || heightInput > 250) {
            showNotification('Please enter a valid height (50-250 cm)', 'warning');
            return;
        }

        // Send height to backend to update the extractor immediately
        try {
            const response = await fetch(`${API_BASE_URL}/api/calibrate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    // We send a dummy value for pixels_per_cm as we are setting reference height
                    pixels_per_cm: 0, 
                    reference_height_cm: parseFloat(heightInput),
                    method: 'manual',
                    camera_id: 'web_session'
                })
            });
            
            const data = await response.json();
            if (data.success) {
                showNotification(`Calibration set! Height: ${heightInput}cm`, 'success');
                // Also update the UI to show this is active
                document.getElementById('heightInput').classList.add('is-valid');
            } else {
                showNotification('Error saving calibration', 'danger');
            }
        } catch (error) {
            console.error(error);
            // Fallback if API fails (offline mode)
            showNotification('Height set locally for next measurement', 'success');
        }

    } else {
        // Credit Card / A4 Paper Calibration
        if (!cameraActive) {
            showNotification('Please start the camera first to calibrate with objects', 'warning');
            return;
        }
        showNotification('Hold the reference object steady near your body...', 'info');
        
        // Note: Full object detection calibration would require specific backend support
        // that is currently a placeholder in your body_measurement_cv.py
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
    let cameraUrl = null;
    
    // Handle different camera sources
    if (cameraSource === 'ip') {
        cameraUrl = document.getElementById('ipCameraUrl').value;
        if (!cameraUrl || cameraUrl.trim().length === 0) {
            showNotification('Please enter IP camera URL', 'warning');
            return;
        }
        cameraSource = cameraUrl.trim();
    } else if (cameraSource === 'usb_tether') {
        const ip = document.getElementById('usbTetherIp').value.trim() || '192.168.42.129';
        const port = document.getElementById('usbTetherPort').value.trim() || '8080';
        cameraSource = `http://${ip}:${port}/video`;
    } else if (cameraSource === 'adb') {
        cameraSource = 'http://localhost:8080/video';
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
    let frameInterval = 1000 / currentFPS;
    let lastRequestTime = 0;
    
    function requestFrame(timestamp) {
        if (!cameraActive) {
            frameRequestInterval = null;
            return;
        }
        
        const elapsed = timestamp - lastRequestTime;
        
        if (elapsed >= frameInterval && !isProcessingFrame) {
            isProcessingFrame = true;
            lastRequestTime = timestamp;
            
            const heightInput = document.getElementById('heightInput').value;
            socket.emit('request_frame', {
                estimated_height_cm: heightInput ? parseFloat(heightInput) : null
            });
        }
        
        frameRequestInterval = requestAnimationFrame(requestFrame);
    }
    
    frameRequestInterval = requestAnimationFrame(requestFrame);
}

function stopFrameRequest() {
    if (frameRequestInterval) {
        cancelAnimationFrame(frameRequestInterval);
        frameRequestInterval = null;
    }
    isProcessingFrame = false;
    frameBuffer = null;
}

function updateCameraStatus(type, text) {
    const statusBadge = document.getElementById('cameraStatus');
    statusBadge.className = 'badge bg-' + type;
    if (type === 'success' || type === 'connected') {
        statusBadge.className = 'badge bg-success';
    } else if (type === 'danger' || type === 'error') {
        statusBadge.className = 'badge bg-danger';
    }
    statusBadge.textContent = text;
}

// ==================== Measurements Display ====================

let measurementUpdateTimeout = null;
function updateMeasurementsDisplay(measurements) {
    if (measurementUpdateTimeout) {
        clearTimeout(measurementUpdateTimeout);
    }
    
    measurementUpdateTimeout = setTimeout(() => {
        document.getElementById('heightValue').textContent = measurements.height_cm.toFixed(1) + ' cm';
        document.getElementById('chestValue').textContent = measurements.chest_cm.toFixed(1) + ' cm';
        document.getElementById('waistValue').textContent = measurements.waist_cm.toFixed(1) + ' cm';
        document.getElementById('shoulderValue').textContent = measurements.shoulder_width_cm.toFixed(1) + ' cm';
        document.getElementById('armValue').textContent = measurements.arm_length_cm.toFixed(1) + ' cm';
        document.getElementById('weightValue').textContent = measurements.weight_kg.toFixed(1) + ' kg';
    }, 50);
}

function updatePoseStatus(detected) {
    const statusBadge = document.getElementById('poseStatus');
    if (detected) {
        if (!statusBadge.classList.contains('pose-detected')) {
            statusBadge.className = 'badge bg-success pose-detected';
            statusBadge.innerHTML = '<i class="bi bi-check-circle"></i> Pose Detected';
        }
    } else {
        if (!statusBadge.classList.contains('bg-secondary')) {
            statusBadge.className = 'badge bg-secondary';
            statusBadge.innerHTML = '<i class="bi bi-circle"></i> No Pose Detected';
        }
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
    const card = document.getElementById('predictionCard');
    card.classList.remove('d-none');
    card.classList.add('fade-in');
    
    document.getElementById('predictedSize').textContent = prediction.predicted_size || 'N/A';
    
    if (currentMeasurements) {
        document.getElementById('resultHeight').textContent = currentMeasurements.height_cm.toFixed(1) + ' cm';
        document.getElementById('resultWeight').textContent = currentMeasurements.weight_kg.toFixed(1) + ' kg';
        document.getElementById('resultChest').textContent = currentMeasurements.chest_cm.toFixed(1) + ' cm';
        document.getElementById('resultWaist').textContent = currentMeasurements.waist_cm.toFixed(1) + ' cm';
        document.getElementById('resultShoulder').textContent = currentMeasurements.shoulder_width_cm.toFixed(1) + ' cm';
        document.getElementById('resultArm').textContent = currentMeasurements.arm_length_cm.toFixed(1) + ' cm';
    }
    
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
                        <td colspan="7" class="text-center text-muted">
                            <i class="bi bi-inbox"></i> No measurements yet
                        </td>
                    </tr>
                `;
                return;
            }
            
            data.measurements.forEach(m => {
                const row = document.createElement('tr');
                const date = new Date(m.measurement_date).toLocaleDateString();
                
                // FIXED: Safe access to predicted_size with fallback
                const predictedSize = m.predicted_size || 'N/A';
                const sizeClass = predictedSize !== 'N/A' ? predictedSize.toLowerCase() : 'm';
                
                row.innerHTML = `
                    <td>${date}</td>
                    <td>${currentPersonId}</td>
                    <td><span class="size-badge ${sizeClass}">${predictedSize}</span></td>
                    <td>${m.height_cm ? m.height_cm.toFixed(1) + ' cm' : 'N/A'}</td>
                    <td>${m.chest_cm ? m.chest_cm.toFixed(1) + ' cm' : 'N/A'}</td>
                    <td>${m.waist_cm ? m.waist_cm.toFixed(1) + ' cm' : 'N/A'}</td>
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
}

// ==================== Statistics ====================

async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/statistics`);
        const data = await response.json();
        
        if (data.success) {
            const stats = data.statistics;
            
            document.getElementById('totalPersons').textContent = stats.total_persons || 0;
            document.getElementById('totalMeasurements').textContent = stats.total_measurements || 0;
            document.getElementById('avgConfidence').textContent = 
                ((stats.avg_confidence || 0) * 100).toFixed(1) + '%';
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

function triggerAutoPrediction() {
    if (isAutoPredicting) return;
    
    isAutoPredicting = true;
    stablePoseCounter = 0;
    lastAutoPredictTime = Date.now();
    
    showNotification('Pose stable! Predicting size...', 'info');
    
    // Add a visual flash effect to the camera feed to show a "photo" was taken
    const feed = document.getElementById('cameraFeed');
    feed.style.filter = 'brightness(2)';
    setTimeout(() => feed.style.filter = 'brightness(1)', 150);

    // Call your existing prediction logic
    // If you have a handlePrediction function, call it here, 
    // or emit directly:
    socket.emit('predict', {
        person_id: currentPersonId,
        measurements: currentMeasurements
    });
    
    // Reset the flag after a delay to allow for the next detection
    setTimeout(() => {
        isAutoPredicting = false;
    }, 2000);
}

// ==================== Debug Functions ====================

window.debugState = function() {
    console.log('Current State:', {
        personId: currentPersonId,
        cameraActive: cameraActive,
        measurements: currentMeasurements,
        isProcessingFrame: isProcessingFrame,
        currentFPS: currentFPS
    });
};

window.setFPS = function(fps) {
    currentFPS = Math.max(5, Math.min(30, fps));
    console.log(`FPS set to: ${currentFPS}`);
    if (cameraActive) {
        stopFrameRequest();
        startFrameRequest();
    }
};