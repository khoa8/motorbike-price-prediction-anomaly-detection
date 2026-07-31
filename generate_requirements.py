from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "artifacts" / "deployment_config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        "Chưa có artifacts/deployment_config.json. "
        "Hãy giải nén bundle vào project trước."
    )

with open(CONFIG_PATH, encoding="utf-8") as file:
    config = json.load(file)

versions = config["versions"]

requirements = [
    # Streamlit sẽ tự cài một phiên bản
    # PyArrow tương thích: >= 7 và < 25.
    "streamlit==1.60.0",

    # Giữ đúng các phiên bản đã dùng để lưu model.
    f"numpy=={versions['numpy']}",
    f"pandas=={versions['pandas']}",
    f"scipy=={versions['scipy']}",
    f"scikit-learn=={versions['scikit_learn']}",
    f"joblib=={versions['joblib']}",
]

output_path = ROOT_DIR / "requirements.txt"
output_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

print("Created:", output_path)
print(output_path.read_text(encoding="utf-8"))
