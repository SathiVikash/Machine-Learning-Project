"""
Municipal Heat Action Plan (HAP) Proposal & Review Service.
Provides persistent SQLite database storage, full lifecycle management (Draft -> Submitted -> Approved -> Completed),
role assignments, reviewer comments, and persistent local database action logs.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from src.config import PROPOSAL_STATUSES, URGENCY_LEVELS, MUNICIPAL_DEPARTMENTS

VALID_TRANSITIONS = {
    "Draft": ["Submitted", "Rejected"],
    "Submitted": ["Approved", "Rejected", "Draft"],
    "Approved": ["Completed", "Rejected"],
    "Rejected": ["Draft"],  # Reopened to Draft for revision only
    "Completed": []         # Terminal state: cannot be reopened or transitioned
}

def sanitize_for_csv(val: Any) -> Any:
    """
    Guards against CSV formula injection (OWASP Spreadsheet Formula Injection).
    Prefixes values starting with =, +, -, @, or tab/return with an apostrophe.
    """
    if isinstance(val, str) and val and val[0] in ('=', '+', '-', '@', '\t', '\r'):
        return f"'{val}"
    return val

class ProposalService:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.db_path = base_dir / "data" / "app_database.db"
        else:
            self.db_path = Path(db_path)
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes tables for proposals and persistent database action audit trails."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Proposals Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_code TEXT UNIQUE NOT NULL,
                    region_key TEXT NOT NULL,
                    ward_id TEXT NOT NULL,
                    ward_name TEXT NOT NULL,
                    intervention_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    department TEXT NOT NULL,
                    urgency TEXT NOT NULL,
                    target_beneficiaries TEXT,
                    budget_inr REAL DEFAULT 0.0,
                    justification TEXT,
                    expected_cooling_c REAL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'Draft',
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Audit Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS proposal_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id INTEGER NOT NULL,
                    previous_status TEXT,
                    new_status TEXT NOT NULL,
                    changed_by TEXT NOT NULL,
                    changed_at TEXT NOT NULL,
                    comment TEXT,
                    FOREIGN KEY(proposal_id) REFERENCES proposals(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

        # Seed sample operational records if database is new
        self.seed_initial_proposals_if_empty()

    def create_proposal(self, data: Dict[str, Any]) -> int:
        """Creates a new Heat Action Proposal and records initial audit log."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate clean human-readable code
        reg_abbr = "TRI" if "trichy" in data.get("region_key", "").lower() else "CHN"
        rand_suffix = datetime.now().strftime("%f")[:3]
        code = f"HAP-{reg_abbr}-{datetime.now().strftime('%Y')}-{rand_suffix}"

        status = data.get("status", "Submitted")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO proposals (
                    proposal_code, region_key, ward_id, ward_name,
                    intervention_type, title, department, urgency,
                    target_beneficiaries, budget_inr, justification,
                    expected_cooling_c, status, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                code,
                data.get("region_key"),
                str(data.get("ward_id")),
                data.get("ward_name"),
                data.get("intervention_type", "General HAP Measure"),
                data.get("title", f"Heat Intervention for Ward {data.get('ward_name')}"),
                data.get("department", MUNICIPAL_DEPARTMENTS[0]),
                data.get("urgency", URGENCY_LEVELS[0]),
                data.get("target_beneficiaries", "Vulnerable urban population"),
                float(data.get("budget_inr", 0.0)),
                data.get("justification", ""),
                float(data.get("expected_cooling_c", 0.0)),
                status,
                data.get("created_by", "Ward Executive Officer"),
                now_str,
                now_str
            ))
            proposal_id = cursor.lastrowid

            # Initial Audit entry
            cursor.execute("""
                INSERT INTO proposal_audit_log (
                    proposal_id, previous_status, new_status, changed_by, changed_at, comment
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                proposal_id,
                None,
                status,
                data.get("created_by", "Ward Executive Officer"),
                now_str,
                "Initial creation and dispatch submission."
            ))
            conn.commit()

        return proposal_id

    def update_proposal_status(self, proposal_id: int, new_status: str, changed_by: str, comment: str) -> bool:
        """
        Transitions proposal to a new status with server-side state machine enforcement
        and logs the reviewer remarks. Completed proposals are terminal.
        """
        if new_status not in PROPOSAL_STATUSES:
            raise ValueError(f"Invalid status: {new_status}. Must be one of {PROPOSAL_STATUSES}")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch current status
            cursor.execute("SELECT status FROM proposals WHERE id = ?", (proposal_id,))
            row = cursor.fetchone()
            if not row:
                return False
            prev_status = row["status"]

            # Server-side transition validation
            allowed_next = VALID_TRANSITIONS.get(prev_status, [])
            if new_status != prev_status and new_status not in allowed_next:
                raise ValueError(
                    f"Illegal state transition from '{prev_status}' to '{new_status}'. "
                    f"Allowed transitions: {allowed_next if allowed_next else 'None (Terminal State)'}."
                )

            # Update proposal
            cursor.execute("""
                UPDATE proposals 
                SET status = ?, updated_at = ?
                WHERE id = ?
            """, (new_status, now_str, proposal_id))

            # Add audit record
            cursor.execute("""
                INSERT INTO proposal_audit_log (
                    proposal_id, previous_status, new_status, changed_by, changed_at, comment
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (proposal_id, prev_status, new_status, changed_by, now_str, comment))

            conn.commit()
            return True

    def export_sanitized_csv(self, region_key: Optional[str] = None) -> bytes:
        """Exports proposals as CSV with formula injection sanitization applied."""
        df = self.get_proposals(region_key=region_key)
        if df.empty:
            return b""
        
        # Sanitize string columns
        sanitized_df = df.copy()
        for col in sanitized_df.select_dtypes(include=['object']).columns:
            sanitized_df[col] = sanitized_df[col].apply(sanitize_for_csv)
            
        return sanitized_df.to_csv(index=False).encode('utf-8')

    def get_proposals(self, region_key: Optional[str] = None, status_filter: Optional[str] = None) -> pd.DataFrame:
        """Fetches proposals as a pandas DataFrame for dashboard rendering."""
        query = "SELECT * FROM proposals WHERE 1=1"
        params = []

        if region_key:
            query += " AND region_key = ?"
            params.append(region_key)

        if status_filter and status_filter != "All":
            query += " AND status = ?"
            params.append(status_filter)

        query += " ORDER BY id DESC"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_proposal_by_id(self, proposal_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single proposal as a dictionary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_audit_trail(self, proposal_id: int) -> List[Dict[str, Any]]:
        """Retrieves chronological review and status history for a proposal."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM proposal_audit_log 
                WHERE proposal_id = ? 
                ORDER BY id ASC
            """, (proposal_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def seed_initial_proposals_if_empty(self):
        """Pre-populates the database with realistic demonstration proposals if table is empty."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM proposals")
            if cursor.fetchone()["cnt"] > 0:
                return  # Already populated

        # Seed realistic Tamil Nadu municipal proposals
        sample_proposals = [
            {
                "region_key": "trichy_delta",
                "ward_id": "W-44",
                "ward_name": "Ward 44 (Palakkarai)",
                "intervention_type": "Emergency ORS Hydration & Shade Kiosks",
                "title": "Palakkarai Market Sector Heat Relief Kiosks",
                "department": "Public Health Department (UPHC)",
                "urgency": "Immediate Response",
                "target_beneficiaries": "Street vendors, cart pullers, and daily-wage laborers",
                "budget_inr": 85000,
                "justification": "Peak surface temperature measured at 43.8°C with elderly ratio > 14%. Immediate hydration required per HAP guidelines.",
                "expected_cooling_c": 1.2,
                "status": "Approved",
                "created_by": "Health Officer (Zone 3)"
            },
            {
                "region_key": "trichy_delta",
                "ward_id": "W-12",
                "ward_name": "Ward 12 (Srirangam)",
                "intervention_type": "Miyawaki Urban Pocket Forestry",
                "title": "North Temple Corridor Canopy Expansion",
                "department": "Parks & Urban Greening Division",
                "urgency": "Seasonal HAP (Pre-Summer)",
                "target_beneficiaries": "Pilgrim transit queues and residential households",
                "budget_inr": 240000,
                "justification": "Extreme vegetation deficit (NDVI 0.11) along pedestrian corridors leading from river crossing.",
                "expected_cooling_c": 2.1,
                "status": "Submitted",
                "created_by": "City Arborist"
            },
            {
                "region_key": "chennai_nondelta",
                "ward_id": "W-55",
                "ward_name": "Ward 55 (George Town / Broadway)",
                "intervention_type": "Solar Reflective Cool Roof Retrofit",
                "title": "Commercial Wholesale Corridor Cool Roof Coating",
                "department": "Engineering & Public Works (PWD)",
                "urgency": "Urgent (1-2 weeks)",
                "target_beneficiaries": "Informal housing, tin-roof residential units",
                "budget_inr": 450000,
                "justification": "Severe building density (78% footprint) and low albedo (0.12) causing severe night-time heat island effect.",
                "expected_cooling_c": 2.8,
                "status": "Completed",
                "created_by": "Superintending Engineer"
            }
        ]

        for p in sample_proposals:
            self.create_proposal(p)

# Global singleton
proposal_service = ProposalService()
