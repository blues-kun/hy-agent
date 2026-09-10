"""Check the public documentation and the intended release file boundaries."""
from pathlib import Path
import re
import subprocess

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
