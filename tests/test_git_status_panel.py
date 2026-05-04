from igris.layers.git_layer.git_status import get_git_info


def test_git_status_returns_fields():
    info = get_git_info()
    # branch and remote may be None if not a git repo
    assert hasattr(info, "dirty")
    assert isinstance(info.changed, list)