from hashlib import blake2b
import hashlib
import re
from pathlib import Path
from data_io import attribute_value, college, USER, TIME, read_history
from older_call_recall import CALL
from newcomer_portrait import stage
BASE=Path(__file__).resolve().parent

def major(value):
    return re.sub(r"本科$","",attribute_value(value)).strip()

def clc(call):
    m = re.match(r"([A-Z]+\d{1,3})",call.upper())
    return m.group(1) if m else call.split("/")[0]

def bucket(value, n):
    return int(hashlib.sha256(value.encode()).hexdigest()[:8],16)%n
def hashed(value,size):
    if not value:
        return 0
    return 1+int.from_bytes(blake2b(value.encode(),digest_size=8).digest(),"little")%(size-1)
