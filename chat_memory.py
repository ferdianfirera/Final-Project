"""
Chat Memory Module - Persistent storage for chat history
This module provides functionality to save and load chat history to/from SQLite database
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class ChatMemory:
    def __init__(self, db_path: str = "chat_history.db"):
        """Initialize chat memory with database connection"""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create chat history table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent TEXT,
                sql_query TEXT,
                query_plan TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def save_message(self, session_id: str, message: Dict):
        """Save a single message to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract metadata
        metadata = {
            "viz_config": message.get("viz_config"),
            "dataframe_shape": None
        }
        
        if "dataframe" in message and message["dataframe"] is not None:
            metadata["dataframe_shape"] = message["dataframe"].shape
        
        cursor.execute("""
            INSERT INTO chat_messages 
            (session_id, role, content, agent, sql_query, query_plan, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            message.get("role"),
            message.get("content"),
            message.get("agent"),
            message.get("sql"),
            message.get("plan"),
            json.dumps(metadata)
        ))
        
        # Update session last_updated
        cursor.execute("""
            UPDATE chat_sessions 
            SET last_updated = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (session_id,))
        
        # Create session if doesn't exist
        if cursor.rowcount == 0:
            cursor.execute("""
                INSERT INTO chat_sessions (session_id)
                VALUES (?)
            """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def load_session(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Load chat history for a specific session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT role, content, agent, sql_query, query_plan, metadata, timestamp
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            msg = {
                "role": row[0],
                "content": row[1],
            }
            if row[2]:  # agent
                msg["agent"] = row[2]
            if row[3]:  # sql_query
                msg["sql"] = row[3]
            if row[4]:  # query_plan
                msg["plan"] = row[4]
            if row[5]:  # metadata
                try:
                    metadata = json.loads(row[5])
                    if metadata.get("viz_config"):
                        msg["viz_config"] = metadata["viz_config"]
                except:
                    pass
            
            messages.append(msg)
        
        return messages
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get list of recent chat sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.session_id, s.created_at, s.last_updated,
                   COUNT(m.id) as message_count,
                   (SELECT content FROM chat_messages 
                    WHERE session_id = s.session_id AND role = 'user'
                    ORDER BY timestamp ASC LIMIT 1) as first_message
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.last_updated DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "created_at": row[1],
                "last_updated": row[2],
                "message_count": row[3],
                "preview": row[4][:50] + "..." if row[4] and len(row[4]) > 50 else row[4]
            })
        
        return sessions
    
    def delete_session(self, session_id: str):
        """Delete a chat session and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary statistics for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_messages,
                SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) as user_messages,
                SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
                SUM(CASE WHEN agent = 'SQL' THEN 1 ELSE 0 END) as sql_queries,
                SUM(CASE WHEN agent = 'RAG' THEN 1 ELSE 0 END) as rag_queries,
                MIN(timestamp) as first_message_time,
                MAX(timestamp) as last_message_time
            FROM chat_messages
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "total_messages": row[0],
                "user_messages": row[1],
                "assistant_messages": row[2],
                "sql_queries": row[3],
                "rag_queries": row[4],
                "first_message_time": row[5],
                "last_message_time": row[6]
            }
        return {}
