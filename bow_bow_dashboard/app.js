let allPets = [];
let allLogs = [];
let isSimulating = false;
let simInterval = null;
let careChartInstance = null;
let zoneChartInstance = null;

const ZONES_LIST = [
  "Room",
  "Hall",
  "Garden",
  "Food Area",
  "Washroom",
  "Out of Camera Range"
];

// Fallback Local Storage Data if API is unavailable
const LOCAL_SAMPLE_PETS = [
  { pet_id: "PET-001", name: "Bruno", breed: "Labrador", age: 4, zone: "Garden", ate: 1, drank: 1, food_grams: 185, water_ml: 320, last_food: "1:10 PM", last_water: "2:05 PM", last_activity: "Just now", alert_status: "Normal" },
  { pet_id: "PET-002", name: "Bella", breed: "Golden Retriever", age: 3, zone: "Room", ate: 1, drank: 0, food_grams: 180, water_ml: 120, last_food: "12:30 PM", last_water: "11:15 AM", last_activity: "5m ago", alert_status: "Needs Water" },
  { pet_id: "PET-003", name: "Rocky", breed: "German Shepherd", age: 5, zone: "Garden", ate: 0, drank: 1, food_grams: 0, water_ml: 410, last_food: "Yesterday", last_water: "1:45 PM", last_activity: "12m ago", alert_status: "Meal Pending" },
  { pet_id: "PET-004", name: "Coco", breed: "Beagle", age: 2, zone: "Food Area", ate: 1, drank: 1, food_grams: 150, water_ml: 360, last_food: "1:15 PM", last_water: "1:50 PM", last_activity: "Just now", alert_status: "Normal" },
  { pet_id: "PET-005", name: "Max", breed: "Indie", age: 6, zone: "Hall", ate: 0, drank: 0, food_grams: 0, water_ml: 80, last_food: "Yesterday", last_water: "10:00 AM", last_activity: "22m ago", alert_status: "Attention" },
  { pet_id: "PET-006", name: "Luna", breed: "Pomeranian", age: 2, zone: "Washroom", ate: 1, drank: 1, food_grams: 90, water_ml: 260, last_food: "11:45 AM", last_water: "12:50 PM", last_activity: "18m ago", alert_status: "Normal" }
];

document.addEventListener("DOMContentLoaded", () => {
  fetchDashboardData();
  fetchLogs();
  setInterval(() => {
    if (isSimulating) {
      triggerSimulationStep();
    }
  }, 3500);
});

async function fetchDashboardData() {
  try {
    const res = await fetch("/api/pets");
    if (res.ok) {
      const data = await res.json();
      allPets = data.pets;
    } else {
      useLocalFallback();
    }
  } catch (err) {
    useLocalFallback();
  }
  renderDashboard();
}

function useLocalFallback() {
  if (allPets.length === 0) {
    allPets = LOCAL_SAMPLE_PETS;
  }
}

async function fetchLogs() {
  try {
    const res = await fetch("/api/logs");
    if (res.ok) {
      const data = await res.json();
      allLogs = data.logs;
      renderLogsTable();
    }
  } catch (err) {
    // Local log fallback
  }
}

function renderDashboard() {
  updateMetrics();
  renderZoneMap();
  applyFilters();
  renderCharts();
}

function updateMetrics() {
  const total = allPets.length;
  const fed = allPets.filter(p => p.ate === 1).length;
  const hydrated = allPets.filter(p => p.drank === 1).length;
  const attention = allPets.filter(p => p.alert_status !== "Normal").length;

  document.getElementById("metric-total").innerText = total;
  document.getElementById("metric-fed").innerText = `${fed}/${total}`;
  document.getElementById("metric-hydrated").innerText = `${hydrated}/${total}`;
  document.getElementById("metric-attention").innerText = attention;

  const fedPct = total > 0 ? Math.round((fed / total) * 100) : 0;
  const hydPct = total > 0 ? Math.round((hydrated / total) * 100) : 0;

  document.getElementById("badge-fed").innerText = `${fedPct}% completed`;
  document.getElementById("badge-hydrated").innerText = `${hydPct}% completed`;
  document.getElementById("badge-attention").innerText = `${attention} need care`;
}

function renderZoneMap() {
  const container = document.getElementById("zone-grid-container");
  container.innerHTML = "";

  ZONES_LIST.forEach(zone => {
    const petsInZone = allPets.filter(p => p.zone === zone);
    const card = document.createElement("div");
    card.className = "zone-card";
    
    let chipsHtml = petsInZone.length === 0 
      ? `<span style="font-size:0.78rem; color:var(--text-muted);">Empty</span>`
      : petsInZone.map(p => `
        <span class="pet-chip">
          <span>🐾</span> ${p.name}
        </span>
      `).join("");

    card.innerHTML = `
      <div class="zone-name">
        <span>${getZoneIcon(zone)} ${zone}</span>
        <span class="metric-badge badge-success">${petsInZone.length}</span>
      </div>
      <div class="zone-pets-chips">${chipsHtml}</div>
    `;
    container.appendChild(card);
  });
}

function getZoneIcon(zone) {
  switch (zone) {
    case "Room": return "🛋️";
    case "Hall": return "🏢";
    case "Garden": return "🌳";
    case "Food Area": return "🥣";
    case "Washroom": return "🧼";
    default: return "📡";
  }
}

function applyFilters() {
  const search = document.getElementById("search-input").value.toLowerCase().trim();
  const zoneVal = document.getElementById("zone-filter").value;
  const statusVal = document.getElementById("status-filter").value;

  const filtered = allPets.filter(p => {
    const matchesSearch = !search || p.name.toLowerCase().includes(search) || p.pet_id.toLowerCase().includes(search) || (p.breed && p.breed.toLowerCase().includes(search));
    const matchesZone = zoneVal === "ALL" || p.zone === zoneVal;
    const matchesStatus = statusVal === "ALL" || p.alert_status === statusVal;
    return matchesSearch && matchesZone && matchesStatus;
  });

  renderPetCards(filtered);
}

function renderPetCards(pets) {
  const container = document.getElementById("pets-container");
  container.innerHTML = "";

  if (pets.length === 0) {
    container.innerHTML = `<div style="grid-column: 1/-1; color:var(--text-muted); text-align:center; padding: 30px;">No pets matching criteria.</div>`;
    return;
  }

  pets.forEach(p => {
    const card = document.createElement("div");
    const statusClass = getStatusCSSClass(p.alert_status);
    card.className = `pet-card ${statusClass}`;

    const foodText = p.ate ? `<span style="color:var(--success); font-weight:700;">✅ Eaten (${p.food_grams}g)</span>` : `<span style="color:var(--warning); font-weight:700;">❌ Pending</span>`;
    const waterText = p.drank ? `<span style="color:var(--success); font-weight:700;">✅ Drank (${p.water_ml}ml)</span>` : `<span style="color:var(--warning); font-weight:700;">❌ Pending</span>`;

    card.innerHTML = `
      <div>
        <div class="pet-card-header">
          <div class="pet-info-header">
            <div class="pet-avatar">${p.name.charAt(0)}</div>
            <div>
              <div class="pet-name-title">${p.name}</div>
              <div class="pet-id-tag">${p.pet_id} · ${p.breed || 'Dog'}</div>
            </div>
          </div>
          <span class="metric-badge ${getBadgeCSSClass(p.alert_status)}">${p.alert_status}</span>
        </div>

        <div class="pet-details-grid">
          <div class="pet-detail-item">
            <span class="detail-label">Current Zone</span>
            <span class="detail-value">${getZoneIcon(p.zone)} ${p.zone}</span>
          </div>
          <div class="pet-detail-item">
            <span class="detail-label">Age</span>
            <span class="detail-value">${p.age} yrs</span>
          </div>
          <div class="pet-detail-item">
            <span class="detail-label">Food Status</span>
            <span class="detail-value">${foodText}</span>
          </div>
          <div class="pet-detail-item">
            <span class="detail-label">Water Status</span>
            <span class="detail-value">${waterText}</span>
          </div>
        </div>
      </div>

      <div class="pet-actions">
        <button class="btn btn-sm btn-secondary" onclick="openActionModal('${p.pet_id}', 'food')">🍲 Feed</button>
        <button class="btn btn-sm btn-secondary" onclick="openActionModal('${p.pet_id}', 'water')">💧 Water</button>
        <button class="btn btn-sm btn-secondary" onclick="openActionModal('${p.pet_id}', 'zone')">📍 Zone</button>
      </div>
    `;
    container.appendChild(card);
  });
}

function getStatusCSSClass(status) {
  switch (status) {
    case "Normal": return "status-normal";
    case "Needs Water": return "status-needs-water";
    case "Meal Pending": return "status-meal-pending";
    default: return "status-attention";
  }
}

function getBadgeCSSClass(status) {
  switch (status) {
    case "Normal": return "badge-success";
    case "Needs Water": return "badge-warning";
    case "Meal Pending": return "badge-warning";
    default: return "badge-danger";
  }
}

function renderLogsTable() {
  const tbody = document.getElementById("logs-body");
  tbody.innerHTML = "";

  allLogs.forEach(log => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="color:var(--text-muted); font-size:0.8rem;">${log.event_time ? log.event_time.split(' ')[1] || log.event_time : ''}</td>
      <td style="font-weight:700;">${log.pet_name}</td>
      <td><span class="pet-chip" style="font-size:0.72rem;">${log.event_type}</span></td>
      <td style="color:var(--text-muted);">${log.details}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderCharts() {
  if (typeof Chart === 'undefined') return;

  const fed = allPets.filter(p => p.ate === 1).length;
  const hydrated = allPets.filter(p => p.drank === 1).length;
  const attention = allPets.filter(p => p.alert_status !== "Normal").length;

  const ctxCare = document.getElementById('careChart').getContext('2d');
  if (careChartInstance) careChartInstance.destroy();
  careChartInstance = new Chart(ctxCare, {
    type: 'doughnut',
    data: {
      labels: ['Fed', 'Hydrated', 'Needs Attention'],
      datasets: [{
        data: [fed, hydrated, attention],
        backgroundColor: ['#34d399', '#38bdf8', '#fbbf24'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8' } },
        title: { display: true, text: 'Care Completion Overview', color: '#f8fafc' }
      }
    }
  });

  const zoneCounts = ZONES_LIST.map(z => allPets.filter(p => p.zone === z).length);
  const ctxZone = document.getElementById('zoneChart').getContext('2d');
  if (zoneChartInstance) zoneChartInstance.destroy();
  zoneChartInstance = new Chart(ctxZone, {
    type: 'bar',
    data: {
      labels: ZONES_LIST,
      datasets: [{
        label: 'Pets in Zone',
        data: zoneCounts,
        backgroundColor: '#a855f7',
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
        y: { ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: '#334155' } }
      },
      plugins: {
        legend: { display: false },
        title: { display: true, text: 'Zone Occupancy Distribution', color: '#f8fafc' }
      }
    }
  });
}

/* Modals & Actions */
function openAddPetModal() {
  document.getElementById("addPetModal").classList.add("active");
}

function closeModal(id) {
  document.getElementById(id).classList.remove("active");
}

async function handleAddPet(e) {
  e.preventDefault();
  const name = document.getElementById("new-name").value;
  const breed = document.getElementById("new-breed").value;
  const age = document.getElementById("new-age").value;
  const zone = document.getElementById("new-zone").value;

  try {
    const res = await fetch("/api/add-pet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, breed, age, zone })
    });
    if (res.ok) {
      closeModal("addPetModal");
      fetchDashboardData();
      fetchLogs();
    }
  } catch (err) {
    // Local update fallback
    const newId = `PET-00${allPets.length + 1}`;
    allPets.push({ pet_id: newId, name, breed, age: parseInt(age), zone, ate: 0, drank: 0, food_grams: 0, water_ml: 0, alert_status: "Attention" });
    closeModal("addPetModal");
    renderDashboard();
  }
}

function openActionModal(petId, type) {
  document.getElementById("action-pet-id").value = petId;
  document.getElementById("action-type").value = type;

  const numContainer = document.getElementById("action-input-container");
  const zoneContainer = document.getElementById("zone-select-container");
  const modalTitle = document.getElementById("action-modal-title");
  const label = document.getElementById("action-label");

  if (type === "food") {
    modalTitle.innerText = "Log Food Consumption (g)";
    label.innerText = "Food Consumed (grams)";
    document.getElementById("action-value-num").value = 150;
    numContainer.style.display = "block";
    zoneContainer.style.display = "none";
  } else if (type === "water") {
    modalTitle.innerText = "Log Water Consumption (ml)";
    label.innerText = "Water Drank (millilitres)";
    document.getElementById("action-value-num").value = 200;
    numContainer.style.display = "block";
    zoneContainer.style.display = "none";
  } else if (type === "zone") {
    modalTitle.innerText = "Update Pet Zone Location";
    numContainer.style.display = "none";
    zoneContainer.style.display = "block";
  }

  document.getElementById("actionModal").classList.add("active");
}

async function handleActionSubmit(e) {
  e.preventDefault();
  const petId = document.getElementById("action-pet-id").value;
  const type = document.getElementById("action-type").value;
  let endpoint = `/api/${type}`;
  let payload = { pet_id: petId };

  if (type === "food") {
    payload.grams = parseInt(document.getElementById("action-value-num").value);
  } else if (type === "water") {
    payload.ml = parseInt(document.getElementById("action-value-num").value);
  } else if (type === "zone") {
    payload.zone = document.getElementById("action-value-zone").value;
  }

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      closeModal("actionModal");
      fetchDashboardData();
      fetchLogs();
    }
  } catch (err) {
    // Local fallback update
    const pet = allPets.find(p => p.pet_id === petId);
    if (pet) {
      if (type === "food") { pet.ate = 1; pet.food_grams += payload.grams; }
      else if (type === "water") { pet.drank = 1; pet.water_ml += payload.ml; }
      else if (type === "zone") { pet.zone = payload.zone; }
      if (pet.ate && pet.drank) pet.alert_status = "Normal";
    }
    closeModal("actionModal");
    renderDashboard();
  }
}

async function resetDailyCare() {
  if (!confirm("Are you sure you want to reset all pets' daily food & water intake counters?")) return;
  try {
    const res = await fetch("/api/reset", { method: "POST" });
    if (res.ok) {
      fetchDashboardData();
      fetchLogs();
    }
  } catch (err) {
    allPets.forEach(p => { p.ate = 0; p.drank = 0; p.food_grams = 0; p.water_ml = 0; p.alert_status = "Attention"; });
    renderDashboard();
  }
}

function toggleSimulation() {
  isSimulating = !isSimulating;
  const btn = document.getElementById("sim-btn");
  if (isSimulating) {
    btn.innerHTML = "⏸️ Pause IoT Simulation";
    btn.classList.add("btn-danger");
    btn.classList.remove("btn-secondary");
  } else {
    btn.innerHTML = "⚡ Start IoT Simulation";
    btn.classList.add("btn-secondary");
    btn.classList.remove("btn-danger");
  }
}

async function triggerSimulationStep() {
  try {
    const res = await fetch("/api/simulate", { method: "POST" });
    if (res.ok) {
      fetchDashboardData();
      fetchLogs();
    }
  } catch (err) {
    // Local simulation fallback
    if (allPets.length > 0) {
      const idx = Math.floor(Math.random() * allPets.length);
      const pet = allPets[idx];
      pet.zone = ZONES_LIST[Math.floor(Math.random() * ZONES_LIST.length)];
      renderDashboard();
    }
  }
}
