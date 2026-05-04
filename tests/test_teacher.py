from igris.core.teacher import build_teacher_payload


def test_teacher_payload_contains_saturated_families():
    tasks = ["run pytest", "run pytest", "run pytest", "write code"]
    payload = build_teacher_payload(tasks, threshold=3)
    assert "testing" in payload["saturated_families"]
    assert len(payload["recent_tasks"]) <= 20