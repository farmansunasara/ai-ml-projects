const API_RECORDS = '/api/records/';
const API_MACHINES = '/api/machines/';

// Store sparkline charts for each machine
const sparklineCharts = new Map();
let distributionChart = null;
let confidenceChart = null;

// Configure Chart.js defaults
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.plugins.legend.display = true;

// Format time for display
function formatTimeAgo(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
}

// Create or update statistics cards
function updateStatistics(machines) {
    const total = machines.length;
    const normal = machines.filter(m => m.predicted_label === 'normal').length;
    const faults = machines.filter(m => m.predicted_label === 'fault').length;
    const avgConfidence = machines.length > 0 
        ? Math.round(machines.reduce((sum, m) => sum + (m.confidence || 0), 0) / machines.length * 100)
        : 0;
    
    document.getElementById('statTotalMachines').textContent = total;
    document.getElementById('statNormal').textContent = normal;
    document.getElementById('statFaults').textContent = faults;
    document.getElementById('statAvgConfidence').textContent = `${avgConfidence}%`;
}

// Initialize sparkline chart
function createSparkline(canvas, data = []) {
    return new Chart(canvas, {
        type: 'line',
        data: {
            labels: new Array(data.length || 10).fill(''),
            datasets: [{
                data: data,
                borderColor: '#0d6efd',
                borderWidth: 2,
                fill: false,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            scales: {
                x: { display: false },
                y: { display: false }
            },
            animation: false,
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            }
        }
    });
}

// Create machine card
function createMachineCard(machine) {
    const template = document.getElementById('machineCardTemplate');
    const clone = template.content.cloneNode(true);
    
    // Set machine ID
    clone.querySelector('.machine-id').textContent = machine.device_id || 'Unknown';
    
    // Set status badge
    const statusBadge = clone.querySelector('.status-badge');
    const isFault = machine.predicted_label && machine.predicted_label.toLowerCase() === 'fault';
    const machineCard = clone.querySelector('.machine-card');
    
    if (isFault) {
        statusBadge.textContent = 'FAULT';
        statusBadge.classList.add('fault');
        machineCard.classList.add('fault-active');
    } else {
        statusBadge.textContent = 'NORMAL';
        statusBadge.classList.add('normal');
        machineCard.classList.add('normal-active');
    }
    
    // Set current metrics
    clone.querySelector('.temperature').textContent = machine.temperature?.toFixed(1) ?? '—';
    clone.querySelector('.vibration').textContent = machine.vibration?.toFixed(1) ?? '—';
    clone.querySelector('.pressure').textContent = machine.pressure?.toFixed(1) ?? '—';
    clone.querySelector('.current').textContent = machine.current?.toFixed(1) ?? '—';
    
    // Set fault prediction info
    clone.querySelector('.fault-type').textContent = isFault ? 'Fault Detected' : 'Normal';
    clone.querySelector('.fault-type').classList.add(isFault ? 'text-danger' : 'text-success');
    
    const confidenceBar = clone.querySelector('.confidence-bar');
    const confidence = machine.confidence ? Math.round(machine.confidence * 100) : 0;
    confidenceBar.style.width = `${confidence}%`;
    confidenceBar.textContent = `${confidence}%`;
    confidenceBar.classList.add(isFault ? 'bg-danger' : 'bg-success');
    
    // Set last update time
    clone.querySelector('.last-update-time').textContent = formatTimeAgo(machine.timestamp);
    
    return clone;
}

// Update sparklines for a machine
function updateSparklines(machineElement, history) {
    const canvases = machineElement.querySelectorAll('.sparkline');
    canvases.forEach(canvas => {
        const metric = canvas.dataset.metric;
        const data = history
            .filter(record => record[metric] != null)
            .slice(-10)
            .map(record => record[metric]);
        
        // Pad with nulls if needed
        while (data.length < 10) data.unshift(null);
        
        let chart = sparklineCharts.get(canvas);
        if (!chart) {
            chart = createSparkline(canvas, data);
            sparklineCharts.set(canvas, chart);
        } else {
            chart.data.datasets[0].data = data;
            chart.update('none');
        }
    });
}

// Update fault distribution chart
function updateDistributionChart(data) {
    const ctx = document.getElementById('faultDistributionChart');
    
    const normalCount = data.filter(d => d.predicted_label === 'normal').length;
    const faultCount = data.filter(d => d.predicted_label === 'fault').length;
    
    if (!distributionChart) {
        distributionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Normal', 'Fault'],
                datasets: [{
                    data: [normalCount, faultCount],
                    backgroundColor: ['#38ef7d', '#eb3349'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    } else {
        distributionChart.data.datasets[0].data = [normalCount, faultCount];
        distributionChart.update();
    }
}

// Update confidence timeline chart
function updateConfidenceChart(data) {
    const ctx = document.getElementById('confidenceChart');
    
    // Get last 20 records
    const recentData = data.slice(-20);
    const labels = recentData.map((_, i) => '');
    const confidenceData = recentData.map(d => (d.confidence || 0) * 100);
    
    if (!confidenceChart) {
        confidenceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence',
                    data: confidenceData,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    } else {
        confidenceChart.data.datasets[0].data = confidenceData;
        confidenceChart.update();
    }
}

// Fetch and display machines with their latest data
async function fetchMachines() {
    const refreshBtn = document.getElementById('refreshBtn');
    const refreshIndicator = refreshBtn.querySelector('.refresh-indicator');
    
    try {
        refreshIndicator.classList.add('active');
        
        // Fetch machines
        const response = await fetch(API_MACHINES);
        if (!response.ok) throw new Error('Failed to fetch machines');
        const machinesData = await response.json();
        const machines = machinesData.machines || [];
        
        // Fetch records for charts and history
        const recordsResponse = await fetch(API_RECORDS);
        if (!recordsResponse.ok) throw new Error('Failed to fetch records');
        const records = await recordsResponse.json();
        
        // Update statistics
        updateStatistics(machines);
        
        // Update distribution chart
        updateDistributionChart(records);
        
        // Update confidence chart
        updateConfidenceChart(records);
        
        // Display machines
        const machineList = document.getElementById('machineList');
        const emptyState = document.getElementById('emptyState');
        
        if (machines.length === 0) {
            machineList.innerHTML = '';
            emptyState.style.display = 'block';
            return;
        }
        
        emptyState.style.display = 'none';
        machineList.innerHTML = '';
        
        // Create and append machine cards
        machines.forEach(machine => {
            const card = createMachineCard(machine);
            machineList.appendChild(card);
        });
        
        // Update sparklines
        setTimeout(() => {
            machineList.querySelectorAll('.card').forEach((machineCard, index) => {
                const machineId = machineCard.querySelector('.machine-id').textContent;
                const history = records
                    .filter(r => r.device_id === machineId)
                    .slice(-10);
                
                updateSparklines(machineCard, history);
            });
        }, 100);
        
    } catch (error) {
        console.error('Error fetching machine data:', error);
    } finally {
        refreshIndicator.classList.remove('active');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set up refresh button
    document.getElementById('refreshBtn').addEventListener('click', fetchMachines);
    
    // Initial fetch
    fetchMachines();
    
    // Auto-refresh every 10 seconds
    setInterval(fetchMachines, 10000);
    
    // Log refresh interval
    console.log('Dashboard initialized. Auto-refresh every 10 seconds.');
});