import json
import sqlite3
import random
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

APP_DIR = Path(__file__).parent.resolve()
DB_PATH = APP_DIR / "bow_bow.db"

ZONES = ["Room", "Hall", "Garden", "Food Area", "Washroom", "Out of Camera Range"]

INITIAL_PETS = [
    ("PET-001", "Bruno", "Labrador", 4, "Garden", 1, 1, 185, 320, "Normal"),
    ("PET-002", "Bella", "Golden Retriever", 3, "Room", 1, 0, 180, 120, "Needs Water"),
    ("PET-003", "Rocky", "German Shepherd", 5, "Garden", 0, 1, 0, 410, "Meal Pending"),
    ("PET-004", "Coco", "Beagle", 2, "Food Area", 1, 1, 150, 360, "Normal"),
    ("PET-005", "Max", "Indie", 6, "Hall", 0, 0, 0, 80, "Attention"),
    ("PET-006", "Luna", "Pomeranian", 2, "Washroom", 1, 1, 90, 260, "Normal"),
]

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
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
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pet_id TEXT NOT NULL,
                pet_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                event_time TEXT NOT NULL
            )
        """)
        
        count = conn.execute("SELECT COUNT(*) FROM pets").fetchone()[0]
        if count == 0:
            now = datetime.now()
            for idx, p in enumerate(INITIAL_PETS):
                pet_id, name, breed, age, zone, ate, drank, food_g, water_ml, status = p
                last_food = (now - timedelta(hours=idx + 1)).strftime("%Y-%m-%d %H:%M:%S") if ate else None
                last_water = (now - timedelta(minutes=(idx + 1) * 35)).strftime("%Y-%m-%d %H:%M:%S") if drank else None
                last_act = (now - timedelta(minutes=(idx + 1) * 12)).strftime("%Y-%m-%d %H:%M:%S")
                
                conn.execute("""
                    INSERT INTO pets (pet_id, name, breed, age, zone, ate, drank, food_grams, water_ml, last_food, last_water, last_activity, alert_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (pet_id, name, breed, age, zone, ate, drank, food_g, water_ml, last_food, last_water, last_act, status))
                
                conn.execute("""
                    INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (pet_id, name, "Location", f"Detected in {zone} via RFID reader", last_act))
        conn.commit()

def calculate_status(ate, drank):
    if not ate and not drank:
        return "Attention"
    if not ate:
        return "Meal Pending"
    if not drank:
        return "Needs Water"
    return "Normal"

class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/pets":
            self.handle_get_pets()
        elif self.path == "/api/logs":
            self.handle_get_logs()
        else:
            super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if self.path == "/api/feed":
            self.handle_feed(data)
        elif self.path == "/api/water":
            self.handle_water(data)
        elif self.path == "/api/zone":
            self.handle_zone(data)
        elif self.path == "/api/simulate":
            self.handle_simulate()
        elif self.path == "/api/add-pet":
            self.handle_add_pet(data)
        elif self.path == "/api/reset":
            self.handle_reset()
        else:
            self.send_json_response({"error": "Endpoint not found"}, status=404)

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def handle_get_pets(self):
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM pets ORDER BY pet_id").fetchall()
            pets = [dict(row) for row in rows]
            self.send_json_response({"pets": pets})

    def handle_get_logs(self):
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM activity_log ORDER BY id DESC LIMIT 100").fetchall()
            logs = [dict(row) for row in rows]
            self.send_json_response({"logs": logs})

    def handle_feed(self, data):
        pet_id = data.get("pet_id")
        grams = int(data.get("grams", 150))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db() as conn:
            pet = conn.execute("SELECT * FROM pets WHERE pet_id = ?", (pet_id,)).fetchone()
            if not pet:
                return self.send_json_response({"error": "Pet not found"}, status=404)
            
            new_ate = 1
            new_status = calculate_status(new_ate, pet["drank"])
            conn.execute("""
                UPDATE pets 
                SET ate = 1, food_grams = food_grams + ?, last_food = ?, last_activity = ?, alert_status = ?
                WHERE pet_id = ?
            """, (grams, now, now, new_status, pet_id))
            
            conn.execute("""
                INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time)
                VALUES (?, ?, 'Feeding', ?, ?)
            """, (pet_id, pet["name"], f"Consumed {grams}g food (Load cell sensor)", now))
            conn.commit()
            self.send_json_response({"message": f"Recorded {grams}g food for {pet['name']}"})

    def handle_water(self, data):
        pet_id = data.get("pet_id")
        ml = int(data.get("ml", 200))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db() as conn:
            pet = conn.execute("SELECT * FROM pets WHERE pet_id = ?", (pet_id,)).fetchone()
            if not pet:
                return self.send_json_response({"error": "Pet not found"}, status=404)
            
            new_drank = 1
            new_status = calculate_status(pet["ate"], new_drank)
            conn.execute("""
                UPDATE pets 
                SET drank = 1, water_ml = water_ml + ?, last_water = ?, last_activity = ?, alert_status = ?
                WHERE pet_id = ?
            """, (ml, now, now, new_status, pet_id))
            
            conn.execute("""
                INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time)
                VALUES (?, ?, 'Hydration', ?, ?)
            """, (pet_id, pet["name"], f"Drank {ml}ml water (Flow sensor)", now))
            conn.commit()
            self.send_json_response({"message": f"Recorded {ml}ml water for {pet['name']}"})

    def handle_zone(self, data):
        pet_id = data.get("pet_id")
        zone = data.get("zone")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db() as conn:
            pet = conn.execute("SELECT * FROM pets WHERE pet_id = ?", (pet_id,)).fetchone()
            if not pet:
                return self.send_json_response({"error": "Pet not found"}, status=404)
            
            conn.execute("""
                UPDATE pets SET zone = ?, last_activity = ? WHERE pet_id = ?
            """, (zone, now, pet_id))
            
            conn.execute("""
                INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time)
                VALUES (?, ?, 'Location', ?, ?)
            """, (pet_id, pet["name"], f"Moved to {zone} (BLE/RFID collar tag)", now))
            conn.commit()
            self.send_json_response({"message": f"Updated zone to {zone} for {pet['name']}"})

    def handle_simulate(self):
        with get_db() as conn:
            pets = conn.execute("SELECT * FROM pets").fetchall()
            if not pets:
                return self.send_json_response({"message": "No pets available"})
            pet = random.choice(pets)
            event_type = random.choice(["zone", "food", "water"])
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if event_type == "zone":
                new_zone = random.choice([z for z in ZONES if z != pet["zone"]])
                conn.execute("UPDATE pets SET zone = ?, last_activity = ? WHERE pet_id = ?", (new_zone, now, pet["pet_id"]))
                conn.execute("INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time) VALUES (?, ?, 'Location', ?, ?)",
                             (pet["pet_id"], pet["name"], f"Simulated IoT move to {new_zone}", now))
                msg = f"IoT Telemetry: {pet['name']} moved to {new_zone}"
            elif event_type == "food":
                g = random.randint(50, 150)
                new_status = calculate_status(1, pet["drank"])
                conn.execute("UPDATE pets SET ate = 1, food_grams = food_grams + ?, last_food = ?, last_activity = ?, alert_status = ? WHERE pet_id = ?",
                             (g, now, now, new_status, pet["pet_id"]))
                conn.execute("INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time) VALUES (?, ?, 'Feeding', ?, ?)",
                             (pet["pet_id"], pet["name"], f"Simulated IoT intake {g}g food", now))
                msg = f"IoT Telemetry: {pet['name']} consumed {g}g food"
            else:
                ml = random.randint(60, 200)
                new_status = calculate_status(pet["ate"], 1)
                conn.execute("UPDATE pets SET drank = 1, water_ml = water_ml + ?, last_water = ?, last_activity = ?, alert_status = ? WHERE pet_id = ?",
                             (ml, now, now, new_status, pet["pet_id"]))
                conn.execute("INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time) VALUES (?, ?, 'Hydration', ?, ?)",
                             (pet["pet_id"], pet["name"], f"Simulated IoT intake {ml}ml water", now))
                msg = f"IoT Telemetry: {pet['name']} drank {ml}ml water"
            
            conn.commit()
            self.send_json_response({"message": msg})

    def handle_add_pet(self, data):
        name = data.get("name")
        breed = data.get("breed", "Mixed")
        age = int(data.get("age", 2))
        zone = data.get("zone", "Room")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM pets").fetchone()[0]
            pet_id = f"PET-{count + 1:03d}"
            
            conn.execute("""
                INSERT INTO pets (pet_id, name, breed, age, zone, ate, drank, food_grams, water_ml, last_food, last_water, last_activity, alert_status)
                VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0, NULL, NULL, ?, 'Attention')
            """, (pet_id, name, breed, age, zone, now))
            
            conn.execute("""
                INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time)
                VALUES (?, ?, 'Registration', ?, ?)
            """, (pet_id, name, f"Registered new pet into resort zone {zone}", now))
            conn.commit()
            self.send_json_response({"message": f"Added pet {name} ({pet_id})"})

    def handle_reset(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_db() as conn:
            conn.execute("""
                UPDATE pets
                SET ate = 0, drank = 0, food_grams = 0, water_ml = 0, last_food = NULL, last_water = NULL, alert_status = 'Attention'
            """)
            conn.execute("""
                INSERT INTO activity_log (pet_id, pet_name, event_type, details, event_time)
                VALUES ('SYSTEM', 'All Pets', 'Daily Reset', 'Daily feeding and hydration counters reset', ?)
            """, (now,))
            conn.commit()
            self.send_json_response({"message": "Daily status reset successfully"})

def run_server(port=8000):
    init_db()
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"[Bow Bow] Pet Resort Dashboard Server running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
