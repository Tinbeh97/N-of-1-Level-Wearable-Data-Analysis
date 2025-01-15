// Token-based access control
const urlParams = new URLSearchParams(window.location.search);
const token = urlParams.get('token');
const validToken = 'n1wearable123'; // Change to your token

if (token !== validToken) {
  document.body.innerHTML = `
    <h1>Access Denied</h1>
    <p>Please provide a valid token to access this analysis.</p>
  `;
} else {
  // Load the analysis content
  loadAnalysis();
}

function loadAnalysis() {
  document.getElementById('content').style.display = 'block';
  loadDataAndRenderCharts();
}

function loadDataAndRenderCharts() {
  fetch('data/wearable_data.json')
    .then(response => response.json())
    .then(data => {
      renderCharts(data);
    })
    .catch(err => console.error('Error loading data:', err));
}
