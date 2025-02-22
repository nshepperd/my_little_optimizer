from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
import sqlite3
import json
from datetime import datetime
import uuid
from contextlib import contextmanager, asynccontextmanager
import threading
from dataclasses import dataclass, asdict
import time


# Database connection management
class Database:
    def __init__(self, db_path: str = "sweeps.db"):
        self.db_path = db_path
        self._connection = None
        self._lock = threading.Lock()

    def _get_connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    @contextmanager
    def get_cursor(self):
        with self._lock:  # Thread safety for SQLite
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None


# Global database instance


# Database initialization
