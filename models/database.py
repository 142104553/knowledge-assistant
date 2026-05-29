"""
SQLite 持久化层

管理：
1. 对话历史（conversations 表）
2. 文档元数据（documents 表，用于去重和删除）
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path


class _JSONEncoder(json.JSONEncoder):
    """处理不可 JSON 序列化的类型（如 datetime、set 等）"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        try:
            return str(obj)
        except:
            return repr(obj)

DB_PATH = Path("./data/app.db")


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = _get_conn()
    cursor = conn.cursor()

    # 对话历史
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_session 
        ON conversations(session_id, created_at)
    """)

    # 文档元数据（用于去重和删除）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════
# 对话历史操作
# ═══════════════════════════════════════════════════════════

def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None
) -> int:
    """保存一条对话消息"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (session_id, role, content, sources) VALUES (?, ?, ?, ?)",
        (session_id, role, content, json.dumps(sources, cls=_JSONEncoder, ensure_ascii=False) if sources else None)
    )
    conn.commit()
    msg_id = cursor.lastrowid
    conn.close()
    return msg_id


def get_conversation_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """获取某个 session 的对话历史"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, sources, created_at FROM conversations WHERE session_id = ? ORDER BY created_at LIMIT ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        sources = json.loads(row["sources"]) if row["sources"] else None
        result.append({
            "role": row["role"],
            "content": row["content"],
            "sources": sources,
            "created_at": row["created_at"]
        })
    return result


def clear_conversation(session_id: str) -> None:
    """清空某个 session 的对话历史"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════
# 文档元数据操作
# ═══════════════════════════════════════════════════════════

def save_document_meta(doc_id: str, filename: str, chunk_count: int = 0) -> None:
    """保存/更新文档元数据"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO documents (doc_id, filename, chunk_count) 
        VALUES (?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET 
            filename=excluded.filename,
            chunk_count=excluded.chunk_count,
            created_at=CURRENT_TIMESTAMP
        """,
        (doc_id, filename, chunk_count)
    )
    conn.commit()
    conn.close()


def get_document_meta(doc_id: str) -> Optional[Dict[str, Any]]:
    """获取文档元数据"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "doc_id": row["doc_id"],
            "filename": row["filename"],
            "chunk_count": row["chunk_count"],
            "created_at": row["created_at"]
        }
    return None


def list_documents() -> List[Dict[str, Any]]:
    """列出所有已上传的文档"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, filename, chunk_count, created_at FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "doc_id": row["doc_id"],
            "filename": row["filename"],
            "chunk_count": row["chunk_count"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]


def delete_document_meta(doc_id: str) -> None:
    """删除文档元数据记录"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
