"""
Ashfall Protocol – Network Protocol
====================================
UDP datagram format:  [MSG_TYPE : 1 byte] [JSON payload : N bytes]

UDP preserves message boundaries, so no length prefix is needed.
JSON is used for readability and ease of debugging.
"""

import json
from enum import IntEnum


class MsgType(IntEnum):
    # ── Lobby phase ────────────────────────────────────────────────
    JOIN        = 0x01   # C→S  {"name": str}
    WELCOME     = 0x02   # S→C  {"pid": int, "players": [...], "ff": bool}
    LOBBY       = 0x03   # S→C  {"players": [{"pid","name","ready"}]}
    READY       = 0x04   # C→S  {}   (toggle)
    START       = 0x05   # S→C  {}
    DISCONNECT  = 0x06   # both {"reason": str}

    # ── Game phase ─────────────────────────────────────────────────
    INPUT       = 0x10   # C→S  {"up","down","left","right","mx","my",
                         #        "shoot","dash","reload","weapon"}
    STATE       = 0x11   # S→C  full snapshot (see ServerGame.serialize_state)
    EVENT       = 0x12   # S→C  {"etype": str, ...}  reliable events

    # ── Utility ────────────────────────────────────────────────────
    PING        = 0x20   # C→S  {"t": float}
    PONG        = 0x21   # S→C  {"t": float}


def pack(msg_type: int, data: dict) -> bytes:
    """Serialize a message to bytes."""
    payload = json.dumps(data, separators=(',', ':')).encode('utf-8')
    return bytes([msg_type]) + payload


def unpack(raw: bytes):
    """Deserialize bytes → (msg_type, data).  Returns (None, None) on error."""
    if not raw:
        return None, None
    try:
        msg_type = raw[0]
        data = json.loads(raw[1:].decode('utf-8'))
        return msg_type, data
    except Exception:
        return None, None
