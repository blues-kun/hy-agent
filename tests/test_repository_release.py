"""Check the public documentation and the intended release file boundaries."""
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_local_links_resolve():
    documents = [ROOT / "README.md", ROOT / "annotation_prelabel/README.md"]
    documents += list((ROOT / "mito").rglob("*.md"))
    documents += list((ROOT / "docs").glob("*.md"))
    documents += list((ROOT / "assets/showcase").glob("*.md"))
    documents += list((ROOT / "assets/training").glob("*.md"))
    missing = []
    for document in documents:
        content = document.read_text(encoding="utf-8")
        # Ignore fenced examples; only real rendered links are checked.
        content = re.sub(r"```.*?```", "", content, flags=re.S)
        targets = re.findall(r"\]\(([^)]+)\)", content)
        targets += re.findall(r'(?:src|href)="([^"]+)"', content)
        for target in targets:
            if target.startswith(("https://", "http://", "mailto:", "data:", "#")):
                continue
            target = target.split("#", 1)[0].strip("<>")
            if target and not (document.parent / target).exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken documentation links:\n" + "\n".join(missing)


def test_showcase_contains_all_original_media():
    media = list((ROOT / "assets/showcase").glob("*.png")) + list((ROOT / "assets/showcase").glob("*.gif"))
    assert len([p for p in media if p.suffix == ".png"]) == 17
    assert len([p for p in media if p.suffix == ".gif"]) == 9
    assert all(0 < p.stat().st_size < 100 * 1024**2 for p in media)
    assert (ROOT / "assets/showcase/digital-human.gif").is_file()


def test_homepage_displays_selected_gifs_and_preserves_original_gallery():
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    # The user replaced the generic knowledge-Q&A GIF with the mechanism view.
    expected = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "assets/showcase").glob("*.gif")
                      if p.name != "knowledge-assistant.gif")

    def gif_images(markup):
        targets = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', markup)
        targets += re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', markup, flags=re.I)
        return sorted(target for target in targets if target.lower().endswith(".gif"))

    assert gif_images(content) == expected
    expanded = re.sub(r"<details\b[^>]*>.*?</details>", "", content, flags=re.S | re.I)
    assert gif_images(expanded) == expected
    assert "knowledge-assistant.gif" not in content
    assert content.count('src="assets/showcase/mechanism-hypotheses.png"') == 1
    gallery = (ROOT / "assets/showcase/README.md").read_text(encoding="utf-8")
    assert gif_images(gallery) == sorted(p.name for p in (ROOT / "assets/showcase").glob("*.gif"))


def test_project_access_and_contact_are_at_the_end():
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    project_url = "https://agent.blueskun.com:8444/"
    assert content.count(project_url) == 1
    last_line = content.strip().splitlines()[-1]
    assert project_url in last_line
    assert "微信" in last_line and "18299228189" in last_line


def test_homepage_prioritizes_training_and_keeps_showcase_compact():
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = ["## 训练路线与专家标注", "## 微调模型", "## 算法与评测", "## 阶段结果", "## 知识图谱与机制发现", "## 实际场景与落地"]
    positions = [content.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "Qwen3-4B-Instruct-2507" in content
    assert "五位医学工作者" in content
    assert "quantitative-report.png" not in content
    assert "查看量化报告" not in content
    for name in ("literature-search.gif", "experiment-planning.png", "digital-human.gif"):
        tag = re.search(r'<img\b[^>]*src="assets/showcase/' + re.escape(name) + r'"[^>]*>', content)
        assert tag is not None, name
        width = re.search(r'width="(\d+)"', tag.group())
        assert width is not None and int(width.group(1)) <= 640, name
    assert len(re.findall(r"^```math$", content, flags=re.M)) >= 5
    ET.parse(ROOT / "assets/training/mito-training-route.svg")
    ET.parse(ROOT / "assets/mito-mechanism-loop.svg")


def test_math_avoids_github_renderer_incompatibilities():
    documents = [ROOT / "README.md", ROOT / "docs/training-and-evaluation.md", ROOT / "mito/rp_grpo/RP_METHOD.md"]
    # GitHub's renderer rejects operatorname and reparses text as HTML before
    # MathJax; a literal '<' can consume the end of a token-prefix subscript.
    for document in documents:
        content = document.read_text(encoding="utf-8")
        for expression in re.findall(r"```math\n(.*?)\n```", content, flags=re.S):
            assert r"\operatorname" not in expression, document
            assert "<" not in expression, document


def test_private_artifacts_are_ignored_but_annotations_are_publishable():
    private = ["mito/models/base/model.safetensors", "mito/rp_grpo/checkpoints/old/adapter_config.json",
               "mito/rp_grpo/data/passages.jsonl", "mito/rp_grpo/data/measurements.csv",
               "mito/audit/raw.jsonl", "mito/runs/example/manifest.json", ".env"]
    public = ["mito/data/train.jsonl", "mito/data/dev.jsonl", "mito/data/MANIFEST.json",
              "mito/data/FORMAT_CONTRACT.json", "mito/models/README.md",
              "mito/rp_grpo/data/README.md", "assets/showcase/digital-human.gif"]
    for name, expected in [(p, 0) for p in private] + [(p, 1) for p in public]:
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", name], cwd=ROOT, check=False)
        assert result.returncode == expected, name
