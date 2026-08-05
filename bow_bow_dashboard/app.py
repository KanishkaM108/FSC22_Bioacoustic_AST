import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "bow_bow.db"

ZONES = [
    "Room",
    "Hall",
    "Garden",
    "Food Area",
    "Washroom",
    "Out of Camera Range",
]

SAMPLE_PETS = [
    ("PET-001", "Bruno", "Labrador", 4, "Garden", 1, 1, 185, 320, "Normal"),
    ("PET-002", "Bella", "Golden Retriever", 3, "Room", 1, 0, 180, 120, "Needs Water"),
    ("PET-003", "Rocky", "German Shepherd", 5, "Garden", 0, 1, 0, 410, "Meal Pending"),
    ("PET-004", "Coco", "Beagle", 2, "Food Area", 1, 1, 150, 360, "Normal"),
    ("PET-005", "Max", "Indie", 6, "Hall", 0, 0, 0, 80, "Attention"),
    ("PET-006", "Luna", "Pomeranian", 2, "Washroom", 1, 1, 90, 260, "Normal"),
]

st.set_page_config(
    page_title="Bow Bow Pet Resort",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
        .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0; color: #38bdf8;}
        .sub-title {color: #94a3b8; margin-top: 0.15rem; margin-bottom: 1.3rem;}
        .pet-card {
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 16px;
            background: #1e293b;
            min-height: 240px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.2);
            color: #f8fafc;
        }
        .pet-name {font-size: 1.35rem; font-weight: 750;}
        .status-ok {
            display: inline-block; padding: 4px 9px; border-radius: 999px;
            background: rgba(52, 211, 153, 0.15); color: #34d399; font-weight: 700; font-size: 0.82rem;
        }
        .status-alert {
            display: inline-block; padding: 4px 9px; border-radius: 999px;
            background: rgba(248, 113, 113, 0.15); color: #f87171; font-weight: 700; font-size: 0.82rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #334155;
            padding: 12px 16px;
            border-radius: 14px;
            background: #1e293b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def initialize_database():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pets (
                pet_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                breed TEXT,
                age INTEGER,
                zone TEXT NOT NULL,
                ate INTEGER NOT NULL DEFAULT 0,
                drank INTEGER NOT NULL DEFAULT 0,
                food_grams INTEGER NOT NULL DEFAULT 0,
                water_ml INTEGER NOT NULL DEFAULT 0,
                last_food TEXT,
                last_water TEXT,
                last_activity TEXT,
                alert_status TEXT NOT NULL DEFAULT 'Normal'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pet_id TEXT NOT NULL,
                pet_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                event_time TEXT NOT NULL
            )
            """
        )

        count = conn.execute("SELECT COUNT(*) FROM pets").fetchone()[0]
        if count == 0:
            now = datetime.now()
            for idx, row in enumerate(SAMPLE_PETS):
                pet_id, name, breed, age, zone, ate, drank, food_g, water_ml, status = row
                last_food = (now - timedelta(hours=idx + 1)).isoformat(timespec="seconds") if ate else None
                last_water = (now - timedelta(minutes=(idx + 1) * 35)).isoformat(timespec="seconds") if drank else None
                last_activity = (now - timedelta(minutes=(idx + 1) * 12)).isoformat(timespec="seconds")
                conn.execute(
                    """
                    INSERT INTO pets (
                        pet_id, name, breed, age, zone, ate, drank,
                        food_grams, water_ml, last_food, last_water,
                        last_activity, alert_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pet_id, name, breed, age, zone, ate, drank, food_g, water_ml, last_food, last_water, last_activity, status),
                )
                conn.execute(
                    """
                    INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time)
                    VALUES (?, ?, 'Location', ?, ?)
                    """,
                    (pet_id, name, f"Detected in {zone} via BLE/RFID tag", last_activity),
                )
        conn.commit()

def load_pets():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM pets ORDER BY pet_id", conn)

def load_activity():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM activity_log ORDER BY datetime(event_time) DESC LIMIT 100", conn)

def calculate_alert(row):
    if int(row["ate"]) == 0 and int(row["drank"]) == 0:
        return "Attention"
    if int(row["ate"]) == 0:
        return "Meal Pending"
    if int(row["drank"]) == 0:
        return "Needs Water"
    return "Normal"

def log_event(conn, pet_id, pet_name, event_type, details):
    conn.execute(
        """
        INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time)
        VALUES (?, ?, ?, ?, ?)
        """,
        (pet_id, pet_name, event_type, details, datetime.now().isoformat(timespec="seconds")),
    )

def update_pet(pet_id, pet_name, action, value=None):
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        if action == "food":
            grams = int(value or 0)
            conn.execute(
                "UPDATE pets SET ate = 1, food_grams = food_grams + ?, last_food = ?, last_activity = ? WHERE pet_id = ?",
                (grams, now, now, pet_id),
            )
            log_event(conn, pet_id, pet_name, "Feeding", f"Consumed {grams}g food (Load cell)")
        elif action == "water":
            ml = int(value or 0)
            conn.execute(
                "UPDATE pets SET drank = 1, water_ml = water_ml + ?, last_water = ?, last_activity = ? WHERE pet_id = ?",
                (ml, now, now, pet_id),
            )
            log_event(conn, pet_id, pet_name, "Hydration", f"Drank {ml}ml water (Flow sensor)")
        elif action == "zone":
            zone = str(value)
            conn.execute(
                "UPDATE pets SET zone = ?, last_activity = ? WHERE pet_id = ?",
                (zone, now, pet_id),
            )
            log_event(conn, pet_id, pet_name, "Location", f"Moved to {zone}")

        row = conn.execute("SELECT * FROM pets WHERE pet_id = ?", (pet_id,)).fetchone()
        columns = [d[0] for d in conn.execute("SELECT * FROM pets LIMIT 0").description]
        current = dict(zip(columns, row))
        alert = calculate_alert(current)
        conn.execute("UPDATE pets SET alert_status = ? WHERE pet_id = ?", (alert, pet_id))
        conn.commit()

def reset_daily_status():
    with get_connection() as conn:
        conn.execute("UPDATE pets SET ate = 0, drank = 0, food_grams = 0, water_ml = 0, last_food = NULL, last_water = NULL, alert_status = 'Attention'")
        conn.execute(
            "INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time) VALUES ('SYSTEM', 'All Pets', 'Daily Reset', 'Reset feeding and hydration counters', ?)",
            (datetime.now().isoformat(timespec="seconds"),),
        )
        conn.commit()

initialize_database()
pets = load_pets()
activity = load_activity()

st.sidebar.title("🐾 Bow Bow")
st.sidebar.caption("Smart Pet Resort Telemetry")

selected_zones = st.sidebar.multiselect("Filter by zone", options=ZONES, default=[])
status_filter = st.sidebar.selectbox("Filter by care status", ["All", "Normal", "Needs Water", "Meal Pending", "Attention"])
search_text = st.sidebar.text_input("Search pet", placeholder="Name, ID or breed")

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh dashboard", use_container_width=True):
    st.rerun()

if st.sidebar.button("🌅 Reset daily food/water", use_container_width=True):
    reset_daily_status()
    st.sidebar.success("Daily status reset.")
    st.rerun()

filtered = pets.copy()
if selected_zones:
    filtered = filtered[filtered["zone"].isin(selected_zones)]
if status_filter != "All":
    filtered = filtered[filtered["alert_status"] == status_filter]
if search_text.strip():
    q = search_text.strip().lower()
    filtered = filtered[
        filtered["name"].str.lower().str.contains(q)
        | filtered["pet_id"].str.lower().str.contains(q)
        | filtered["breed"].fillna("").str.lower().str.contains(q)
    ]

st.markdown('<p class="main-title">🐾 Bow Bow Pet Resort Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-title">Real-time zone tracking, feeding, hydration and telemetry · Updated {datetime.now().strftime("%I:%M:%S %p")}</p>',
    unsafe_allow_html=True,
)

total_pets = len(pets)
fed_count = int(pets["ate"].sum())
water_count = int(pets["drank"].sum())
alert_count = int((pets["alert_status"] != "Normal").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Pets", total_pets)
m2.metric("Fed Today", f"{fed_count}/{total_pets}", f"{fed_count / max(total_pets, 1) * 100:.0f}%")
m3.metric("Hydrated Today", f"{water_count}/{total_pets}", f"{water_count / max(total_pets, 1) * 100:.0f}%")
m4.metric("Attention Required", alert_count, delta=f"-{alert_count}", delta_color="inverse")

tab_overview, tab_pets, tab_update, tab_activity = st.tabs(
    ["📊 Overview", "🐶 Pet Monitor", "✏️ Staff Actions", "🕒 Activity Log"]
)

with tab_overview:
    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Current Zone Occupancy")
        zone_counts = pets.groupby("zone", as_index=False).size().rename(columns={"size": "pets"}).sort_values("pets", ascending=False)
        zone_chart = px.bar(zone_counts, x="zone", y="pets", text="pets", color="zone", labels={"zone": "Resort Zone", "pets": "Pet Count"})
        zone_chart.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
        st.plotly_chart(zone_chart, use_container_width=True)

    with right:
        st.subheader("Care Completion Overview")
        care_df = pd.DataFrame({
            "Care Item": ["Food Completed", "Water Completed", "Needs Attention"],
            "Pets": [fed_count, water_count, alert_count],
        })
        care_chart = px.pie(care_df, names="Care Item", values="Pets", hole=0.55, color_discrete_sequence=["#34d399", "#38bdf8", "#fbbf24"])
        care_chart.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark")
        st.plotly_chart(care_chart, use_container_width=True)

    st.subheader("Active Care Alerts")
    alerts = pets[pets["alert_status"] != "Normal"][["pet_id", "name", "zone", "alert_status", "last_activity"]].copy()
    if alerts.empty:
        st.success("All pets are happy and care is complete!")
    else:
        alerts.columns = ["Pet ID", "Pet Name", "Current Zone", "Alert Status", "Last Activity"]
        st.dataframe(alerts, use_container_width=True, hide_index=True)

with tab_pets:
    st.subheader(f"Pet Monitoring Cards ({len(filtered)})")
    if filtered.empty:
        st.info("No pets match the current filter selection.")
    else:
        records = filtered.to_dict("records")
        for start in range(0, len(records), 3):
            cols = st.columns(3)
            for col, pet in zip(cols, records[start:start + 3]):
                with col:
                    st.markdown(
                        f"""
                        <div class="pet-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="pet-name">🐾 {pet['name']}</span>
                                <span class="{'status-ok' if pet['alert_status'] == 'Normal' else 'status-alert'}">{pet['alert_status']}</span>
                            </div>
                            <div style="margin-top:6px; font-size:0.85rem; color:#94a3b8;">
                                ID: <b>{pet['pet_id']}</b> · Breed: <b>{pet['breed']}</b>
                            </div>
                            <hr style="border-color:#334155; margin:10px 0;">
                            <div>📍 Zone: <b>{pet['zone']}</b></div>
                            <div>🍲 Food: <b>{'✅ ' + str(pet['food_grams']) + 'g' if pet['ate'] else '❌ Pending'}</b></div>
                            <div>💧 Water: <b>{'✅ ' + str(pet['water_ml']) + 'ml' if pet['drank'] else '❌ Pending'}</b></div>
                            <div style="margin-top:8px; font-size:0.78rem; color:#6b7280;">Last Activity: {pet['last_activity']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

with tab_update:
    st.subheader("Manual Staff Telemetry Override")
    pet_names = pets["name"].tolist()
    if pet_names:
        selected_pet_name = st.selectbox("Select Pet", pet_names)
        pet_row = pets[pets["name"] == selected_pet_name].iloc[0]
        pet_id = pet_row["pet_id"]
        c1, c2, c3 = st.columns(3)
        with c1:
            feed_g = st.number_input("Food intake (grams)", min_value=10, max_value=500, value=150)
            if st.button("Log Feeding"):
                update_pet(pet_id, selected_pet_name, "food", feed_g)
                st.success(f"Logged {feed_g}g food for {selected_pet_name}")
                st.rerun()
        with c2:
            water_ml = st.number_input("Water intake (ml)", min_value=10, max_value=1000, value=200)
            if st.button("Log Hydration"):
                update_pet(pet_id, selected_pet_name, "water", water_ml)
                st.success(f"Logged {water_ml}ml water for {selected_pet_name}")
                st.rerun()
        with c3:
            new_z = st.selectbox("New Zone Location", ZONES)
            if st.button("Update Location"):
                update_pet(pet_id, selected_pet_name, "zone", new_z)
                st.success(f"Moved {selected_pet_name} to {new_z}")
                st.rerun()

with tab_activity:
    st.subheader("Real-time Event & Telemetry Logs")
    st.dataframe(activity[["event_time", "pet_id", "pet_name", "event_type", "details"]], use_container_width=True, hide_index=True)
