from uuid import uuid4


def new_trace_id(prefix: str = "trc") -> str:
    return f"{prefix}_{uuid4().hex}"
