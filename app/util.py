
import time, secrets

def now_ms() -> int:
    return int(time.time() * 1000)

def new_id(prefix: str = "") -> str:
    return prefix + secrets.token_hex(8)
