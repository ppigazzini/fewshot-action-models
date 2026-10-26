import subprocess
import sys
from pathlib import Path

TOOL = (sys.executable, str(Path(__file__).parent.parent / "tools" / "check_markdown.py"))


def _check(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*TOOL, *map(str, paths)], capture_output=True, text=True, check=False)


class TestLinks:
    def test_an_anchor_no_heading_has_is_dead(self, tmp_path):
        page = tmp_path / "page.md"
        page.write_text("# The `relevant` grounding\n\n[a](#the-relevant-grounding) [b](#gone)\n")
        out = _check(page)
        assert out.returncode == 1
        assert "dead link -> #gone" in out.stdout
        assert "#the-relevant-grounding" not in out.stdout

    def test_an_anchor_in_another_page_is_resolved_there(self, tmp_path):
        (tmp_path / "other.md").write_text("## Commit messages\n")
        page = tmp_path / "page.md"
        page.write_text("[a](other.md#commit-messages) [b](other.md#commits)\n")
        out = _check(page)
        assert "dead link -> other.md#commits" in out.stdout
        assert "other.md#commit-messages" not in out.stdout

    def test_a_link_inside_code_is_not_a_link(self, tmp_path):
        page = tmp_path / "page.md"
        page.write_text("`d[k](x)` is code\n\n```python\nf[0](y)\n```\n")
        assert _check(page).returncode == 0

    def test_a_missing_file_is_a_problem(self, tmp_path):
        out = _check(tmp_path / "absent.md")
        assert out.returncode == 1
        assert "absent.md: no such file" in out.stdout
