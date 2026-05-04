from igris.core import anti_loop


def test_saturated_families_detected():
    tasks = [
        "run pytest",
        "write new function",
        "run pytest",
        "run pytest",
        "run pytest",
    ]
    counts = anti_loop.compute_family_counts(tasks)
    sat = anti_loop.saturated_families(counts, threshold=3)
    assert "testing" in sat