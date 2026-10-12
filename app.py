import os
import io
import csv
import uuid
import base64
import sqlite3
import threading
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import (Flask, render_template, request, redirect, url_for,
                   session, jsonify, Response, g)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

load_dotenv(override=True)

# Same proxy clean-up as the original scripts
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
            "ALL_PROXY", "NO_PROXY", "no_proxy"]:
    os.environ.pop(var, None)

from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from qdrant_client.http import models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
os.makedirs(UPLOAD_DIR, exist_ok=True)

COLLECTION_NAME = "student_faces"
SIMILARITY_THRESHOLD = 0.40

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# ---------------- QDRANT ----------------
print("Connecting to Qdrant...")
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60,
    trust_env=False,
    check_compatibility=False,
)
existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )
try:
    qdrant.create_payload_index(COLLECTION_NAME, "student_id",
                                models.PayloadSchemaType.KEYWORD)
except Exception:
    pass
print("Qdrant ready")

# ---------------- INSIGHTFACE ----------------
print("Loading InsightFace...")
face_analyzer = FaceAnalysis(name="antelopev2", root="model",
                             providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
face_lock = threading.Lock()
print("InsightFace ready")


# ---------------- SQLITE (teachers + attendance) ----------------
def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    conn = g.pop("db", None)
    if conn:
        conn.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS teachers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS sessions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        teacher_id INTEGER,
        started_at TEXT NOT NULL,
        ended_at TEXT);
    CREATE TABLE IF NOT EXISTS records(
        session_id INTEGER NOT NULL,
        student_id TEXT NOT NULL,
        name TEXT, department TEXT, year TEXT,
        similarity REAL,
        marked_at TEXT NOT NULL,
        PRIMARY KEY(session_id, student_id));
    """)
    if conn.execute("SELECT COUNT(*) FROM teachers").fetchone()[0] == 0:
        user = os.getenv("DEFAULT_TEACHER", "teacher")
        pw = os.getenv("DEFAULT_PASSWORD", "teacher123")
        conn.execute("INSERT INTO teachers(username,password_hash) VALUES(?,?)",
                     (user, generate_password_hash(pw)))
        print(f"Created first teacher account: {user}")
    conn.commit()
    conn.close()


init_db()


# ---------------- HELPERS ----------------
def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "teacher_id" not in session:
            if request.path.startswith("/api/"):
                return jsonify(error="Not logged in"), 401
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper


def decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def all_students():
    """Unique students (by student_id) from Qdrant."""
    students, offset = {}, None
    while True:
        points, offset = qdrant.scroll(COLLECTION_NAME, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in points:
            pl = p.payload or {}
            sid = pl.get("student_id")
            if sid and sid not in students:
                students[sid] = {
                    "student_id": sid,
                    "name": pl.get("name", ""),
                    "department": pl.get("department", ""),
                    "year": pl.get("year", ""),
                    "registered_at": pl.get("registered_at", ""),
                }
        if offset is None:
            break
    return sorted(students.values(), key=lambda s: s["student_id"])


def now():
    return datetime.now().isoformat(timespec="seconds")


# ---------------- AUTH ----------------
@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("dashboard" if "teacher_id" in session else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        row = db().execute("SELECT * FROM teachers WHERE username=?", (u,)).fetchone()
        if row and check_password_hash(row["password_hash"], p):
            session.clear()
            session["teacher_id"] = row["id"]
            session["username"] = row["username"]
            return redirect(url_for("dashboard"))
        error = "Invalid username or password"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- PAGES ----------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", active="dashboard")


@app.route("/attendance")
@login_required
def attendance_page():
    return render_template("attendance.html", active="attendance",
                           threshold=SIMILARITY_THRESHOLD)


@app.route("/register")
@login_required
def register_page():
    return render_template("register.html", active="register")


@app.route("/reports")
@login_required
def reports_page():
    return render_template("reports.html", active="reports")


# ---------------- API: STATS / STUDENTS ----------------
@app.route("/api/stats")
@login_required
def api_stats():
    students = all_students()
    by_dept = {}
    for s in students:
        k = s["department"] or "—"
        by_dept[k] = by_dept.get(k, 0) + 1
    conn = db()
    n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    today_sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE started_at LIKE ?", (today + "%",)
    ).fetchone()[0]
    return jsonify(total_students=len(students), by_department=by_dept,
                   total_sessions=n_sessions, sessions_today=today_sessions)


@app.route("/api/students")
@login_required
def api_students():
    return jsonify(all_students())


@app.route("/api/students/<student_id>", methods=["DELETE"])
@login_required
def api_delete_student(student_id):
    qdrant.delete(
        COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="student_id",
                                  match=models.MatchValue(value=student_id))])),
    )
    return jsonify(ok=True)


# ---------------- API: REGISTER ----------------
@app.route("/api/register", methods=["POST"])
@login_required
def api_register():
    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    department = request.form.get("department", "").strip()
    year = request.form.get("year", "").strip()
    file = request.files.get("image")

    if not (student_id and name and department and year):
        return jsonify(error="All fields are required"), 400
    if not file:
        return jsonify(error="Please capture or upload a photo"), 400

    dup = qdrant.scroll(
        COLLECTION_NAME,
        scroll_filter=models.Filter(must=[models.FieldCondition(
            key="student_id", match=models.MatchValue(value=student_id))]),
        limit=1)
    if dup[0]:
        return jsonify(error="Student already registered"), 409

    raw = file.read()
    img = decode_image(raw)
    if img is None:
        return jsonify(error="Cannot read image"), 400

    with face_lock:
        faces = face_analyzer.get(img)
    if not faces:
        return jsonify(error="No face detected. Use a clear, front-facing photo."), 422

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = face.embedding.astype(np.float32)

    # Block the same face being registered under a second ID
    res = qdrant.query_points(COLLECTION_NAME, query=embedding.tolist(),
                              limit=1, with_payload=True)
    if res.points and res.points[0].score >= SIMILARITY_THRESHOLD:
        other = res.points[0].payload or {}
        return jsonify(error=f"This face is already registered as "
                             f"{other.get('name')} ({other.get('student_id')})"), 409

    filename = secure_filename(f"{student_id}_{uuid.uuid4().hex[:8]}.jpg")
    path = os.path.join(UPLOAD_DIR, filename)
    cv2.imwrite(path, img)

    qdrant.upsert(COLLECTION_NAME, points=[models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={"student_id": student_id, "name": name, "department": department,
                 "year": year, "image_path": path, "registered_at": now()},
    )])
    return jsonify(ok=True, warning="Multiple faces in photo; used the largest."
                   if len(faces) > 1 else None)


# ---------------- API: ATTENDANCE ----------------
@app.route("/api/session/start", methods=["POST"])
@login_required
def api_session_start():
    title = (request.json or {}).get("title", "").strip() or \
        f"Class {datetime.now().strftime('%d %b %Y %H:%M')}"
    cur = db().execute(
        "INSERT INTO sessions(title,teacher_id,started_at) VALUES(?,?,?)",
        (title, session["teacher_id"], now()))
    db().commit()
    return jsonify(session_id=cur.lastrowid, title=title)


@app.route("/api/session/<int:sid>/end", methods=["POST"])
@login_required
def api_session_end(sid):
    db().execute("UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL",
                 (now(), sid))
    db().commit()
    return jsonify(ok=True)


@app.route("/api/recognize", methods=["POST"])
@login_required
def api_recognize():
    sid = request.form.get("session_id", type=int)
    file = request.files.get("frame")
    if not sid or not file:
        return jsonify(error="session_id and frame required"), 400
    img = decode_image(file.read())
    if img is None:
        return jsonify(error="Bad frame"), 400

    with face_lock:
        faces = face_analyzer.get(img)

    out = []
    conn = db()
    for face in faces:
        emb = face.embedding.astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        item = {"box": [x1, y1, x2, y2], "known": False}
        try:
            res = qdrant.query_points(COLLECTION_NAME, query=emb.tolist(),
                                      limit=1, with_payload=True)
        except Exception:
            out.append(item)
            continue
        if res.points and res.points[0].score >= SIMILARITY_THRESHOLD:
            m = res.points[0]
            pl = m.payload or {}
            item.update(known=True, student_id=pl.get("student_id"),
                        name=pl.get("name"), department=pl.get("department", ""),
                        year=pl.get("year", ""), similarity=round(m.score, 3))
            conn.execute(
                "INSERT OR IGNORE INTO records(session_id,student_id,name,department,"
                "year,similarity,marked_at) VALUES(?,?,?,?,?,?,?)",
                (sid, pl.get("student_id"), pl.get("name"), pl.get("department"),
                 pl.get("year"), float(m.score), now()))
        out.append(item)
    conn.commit()
    present = conn.execute("SELECT COUNT(*) FROM records WHERE session_id=?",
                           (sid,)).fetchone()[0]
    return jsonify(faces=out, present=present, width=img.shape[1], height=img.shape[0])


@app.route("/api/session/<int:sid>/present")
@login_required
def api_session_present(sid):
    rows = db().execute("SELECT student_id,name,department,year,marked_at FROM records "
                        "WHERE session_id=? ORDER BY marked_at", (sid,)).fetchall()
    return jsonify([dict(r) for r in rows])


# ---------------- API: REPORTS ----------------
@app.route("/api/reports")
@login_required
def api_reports():
    rows = db().execute("""
        SELECT s.id, s.title, s.started_at, s.ended_at, t.username AS teacher,
               (SELECT COUNT(*) FROM records r WHERE r.session_id=s.id) AS present
        FROM sessions s LEFT JOIN teachers t ON t.id=s.teacher_id
        ORDER BY s.started_at DESC""").fetchall()
    total = len(all_students())
    return jsonify(total_students=total, sessions=[dict(r) for r in rows])


def session_detail(sid):
    conn = db()
    s = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if not s:
        return None
    present = [dict(r) for r in conn.execute(
        "SELECT student_id,name,department,year,similarity,marked_at FROM records "
        "WHERE session_id=? ORDER BY student_id", (sid,))]
    present_ids = {p["student_id"] for p in present}
    absent = [st for st in all_students() if st["student_id"] not in present_ids]
    return {"session": dict(s), "present": present, "absent": absent}


@app.route("/api/reports/<int:sid>")
@login_required
def api_report_detail(sid):
    d = session_detail(sid)
    return (jsonify(d) if d else (jsonify(error="Not found"), 404))


@app.route("/api/reports/<int:sid>/csv")
@login_required
def api_report_csv(sid):
    d = session_detail(sid)
    if not d:
        return "Not found", 404
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Student ID", "Name", "Department", "Year", "Status", "Marked at"])
    for p in d["present"]:
        w.writerow([p["student_id"], p["name"], p["department"], p["year"],
                    "Present", p["marked_at"]])
    for a in d["absent"]:
        w.writerow([a["student_id"], a["name"], a["department"], a["year"], "Absent", ""])
    fname = secure_filename(f"attendance_{d['session']['title']}_{sid}.csv")
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.route("/api/reports/<int:sid>", methods=["DELETE"])
@login_required
def api_report_delete(sid):
    conn = db()
    conn.execute("DELETE FROM records WHERE session_id=?", (sid,))
    conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
    conn.commit()
    return jsonify(ok=True)


if __name__ == "__main__":
    # getUserMedia (webcam) works on http://localhost; use HTTPS if hosting remotely.
    # app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
