"""Verify the refined pathology pipeline end-to-end via the live FastAPI."""
import base64, json, requests
from pathlib import Path

images = [
    "dataset/chest/person100_bacteria_480.jpeg",
    "dataset/chest/person100_bacteria_475.jpeg",
    "test_images/test.jpg",
]
for i, p in enumerate(images):
    r = requests.post(
        "http://localhost:8000/detect-pathology",
        files={"file": open(p, "rb")},
    )
    d = r.json()
    print(f"\n=== {p} ===")
    print("status:", d.get("status"))
    for patho in d.get("pathologies", []):
        print(f"  [{patho.get('severity'):<8}] {patho['confidence']*100:5.1f}%  {patho.get('description')}")
    summary = d.get("summary") or {}
    if summary.get("primary_finding"):
        print("  SUMMARY:", summary["primary_finding"])
    if summary.get("secondary_finding"):
        print("           ", summary["secondary_finding"])
    print("  bbox:", d.get("bbox"))
    if d.get("disclaimer"):
        print("  disclaimer:", d["disclaimer"][:80], "...")
    hb = d.get("heatmap")
    if hb:
        out = Path(f"test_refined_{i}.png")
        out.write_bytes(base64.b64decode(hb))
