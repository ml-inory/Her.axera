from app.core.tracing import new_trace_id


def test_new_trace_id_has_prefix() -> None:
    tid = new_trace_id()
    assert tid.startswith("trc_"), f"Expected 'trc_' prefix, got: {tid}"


def test_new_trace_id_custom_prefix() -> None:
    tid = new_trace_id(prefix="abc")
    assert tid.startswith("abc_"), f"Expected 'abc_' prefix, got: {tid}"


def test_new_trace_id_is_unique() -> None:
    ids = {new_trace_id() for _ in range(100)}
    assert len(ids) == 100, "Trace IDs should be unique"


def test_new_trace_id_length() -> None:
    tid = new_trace_id()
    assert len(tid) == 36, f"Expected 36 chars, got {len(tid)}: {tid}"
