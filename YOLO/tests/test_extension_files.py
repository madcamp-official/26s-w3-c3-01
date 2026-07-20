from __future__ import annotations

import json
from pathlib import Path
import unittest


EXTENSION = Path(__file__).resolve().parents[1] / "extension"


class ChromeExtensionFilesTest(unittest.TestCase):
    def test_manifest_references_existing_local_files(self) -> None:
        manifest = json.loads((EXTENSION / "manifest.json").read_text(encoding="utf-8"))
        referenced = [
            manifest["background"]["service_worker"],
            manifest["side_panel"]["default_path"],
            *manifest["content_scripts"][0]["js"],
        ]
        for relative_path in referenced:
            self.assertTrue((EXTENSION / relative_path).is_file(), relative_path)

    def test_manifest_has_required_youtube_permissions(self) -> None:
        manifest = json.loads((EXTENSION / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["manifest_version"], 3)
        self.assertIn("sidePanel", manifest["permissions"])
        self.assertIn("http://127.0.0.1:8765/*", manifest["host_permissions"])
        self.assertIn(
            "https://www.youtube.com/*",
            manifest["content_scripts"][0]["matches"],
        )


if __name__ == "__main__":
    unittest.main()
