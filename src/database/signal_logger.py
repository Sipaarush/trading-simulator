"""
Signal Logger Module — Phase 5.

SQLite-based logging for all trade signals.
    - log(trade) -> int (row id)
    - get_recent(limit) -> list
    - get_stats() -> {total, wins, win_rate, avg_confidence, avg_rr}
"""

import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SignalLogger:
    """
    SQLite CRUD for trade signal logging.

    Stores all TradeSignal fields + timestamp + outcome + pnl.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the signal logger.

        Args:
            config: Full config dict from config.yaml.
        """
        db_cfg = config.get('database', {})
        self.db_path = db_cfg.get('path', 'signals.db')
        self.table_name = db_cfg.get('table_name', 'trade_signals')

        self._init_db()
        logger.info("SignalLogger initialized — db=%s", self.db_path)

    def _init_db(self) -> None:
        """Create the signals table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL,
                    entry REAL,
                    stop_loss REAL,
                    target1 REAL,
                    target2 REAL,
                    risk_pct REAL,
                    rr_ratio REAL,
                    position_size REAL,
                    success_pct REAL,
                    pattern_name TEXT,
                    contributing TEXT,
                    atr REAL,
                    outcome TEXT DEFAULT 'pending',
                    pnl REAL DEFAULT 0.0
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to initialize database: %s", e)

    def log(self, trade: object) -> int:
        """
        Log a trade signal to the database.

        Args:
            trade: TradeSignal dataclass.

        Returns:
            Row ID of the inserted record, or -1 on failure.
        """
        try:
            direction = getattr(trade, 'direction', 'unknown')
            confidence = getattr(trade, 'confidence', 0.0)
            entry = getattr(trade, 'entry', 0.0)
            stop_loss = getattr(trade, 'stop_loss', 0.0)
            target1 = getattr(trade, 'target1', 0.0)
            target2 = getattr(trade, 'target2', 0.0)
            risk_pct = getattr(trade, 'risk_pct', 0.0)
            rr_ratio = getattr(trade, 'rr_ratio', 0.0)
            position_size = getattr(trade, 'position_size', 0.0)
            success_pct = getattr(trade, 'success_pct', 0.0)
            pattern_name = getattr(trade, 'pattern_name', 'Unknown')
            contributing = getattr(trade, 'contributing', [])
            atr = getattr(trade, 'atr', 0.0)

            contributing_str = ",".join(contributing) if isinstance(contributing, list) else str(contributing)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {self.table_name}
                (timestamp, direction, confidence, entry, stop_loss,
                 target1, target2, risk_pct, rr_ratio, position_size,
                 success_pct, pattern_name, contributing, atr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(), direction, confidence, entry, stop_loss,
                target1, target2, risk_pct, rr_ratio, position_size,
                success_pct, pattern_name, contributing_str, atr,
            ))
            conn.commit()
            row_id = cursor.lastrowid
            conn.close()

            logger.info("Signal logged — id=%d, direction=%s", row_id, direction)
            return row_id

        except Exception as e:
            logger.error("Failed to log signal: %s", e)
            return -1

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get the most recent signals.

        Args:
            limit: Maximum number of signals to return.

        Returns:
            List of signal dicts, most recent first.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {self.table_name} ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error("Failed to get recent signals: %s", e)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated signal statistics.

        Returns:
            dict with: total, wins, win_rate, avg_confidence, avg_rr
        """
        stats = {
            'total': 0,
            'wins': 0,
            'win_rate': 0.0,
            'avg_confidence': 0.0,
            'avg_rr': 0.0,
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total signals
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            stats['total'] = cursor.fetchone()[0]

            # Wins
            cursor.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE outcome = 'win'"
            )
            stats['wins'] = cursor.fetchone()[0]

            # Win rate
            if stats['total'] > 0:
                stats['win_rate'] = stats['wins'] / stats['total'] * 100

            # Average confidence
            cursor.execute(
                f"SELECT AVG(confidence) FROM {self.table_name}"
            )
            avg_conf = cursor.fetchone()[0]
            stats['avg_confidence'] = avg_conf if avg_conf else 0.0

            # Average R:R
            cursor.execute(
                f"SELECT AVG(rr_ratio) FROM {self.table_name}"
            )
            avg_rr = cursor.fetchone()[0]
            stats['avg_rr'] = avg_rr if avg_rr else 0.0

            conn.close()

        except Exception as e:
            logger.error("Failed to get stats: %s", e)

        return stats

    def update_outcome(
        self,
        signal_id: int,
        outcome: str,
        pnl: float = 0.0,
    ) -> bool:
        """
        Update the outcome of a signal.

        Args:
            signal_id: Row ID of the signal.
            outcome: 'win', 'loss', or 'pending'.
            pnl: Profit/loss amount.

        Returns:
            True if updated successfully.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE {self.table_name} SET outcome = ?, pnl = ? WHERE id = ?",
                (outcome, pnl, signal_id),
            )
            conn.commit()
            conn.close()
            logger.info("Signal %d outcome updated: %s, pnl=%.2f", signal_id, outcome, pnl)
            return True

        except Exception as e:
            logger.error("Failed to update outcome: %s", e)
            return False
