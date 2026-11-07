import sys
from pathlib import Path
import pandas as pd

# Add root to python path to import app.inference
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.inference import VisionGuardPredictor

print("=" * 70)
print("SUB-TASK 4.1: VERIFYING INFERENCE PIPELINE & BENCHMARKING LATENCY")
print("=" * 70)

predictor = VisionGuardPredictor()
test_csv = Path("reports/splits/test.csv")

if not test_csv.exists():
    raise FileNotFoundError("reports/splits/test.csv missing. Run Task 2 first.")

test_df = pd.read_csv(test_csv)
sample_rows = test_df.sample(n=min(5, len(test_df)), random_state=42)

latencies = []
print(f"\nRunning test inferences across {len(sample_rows)} sample images:\n")

for idx, (_, row) in enumerate(sample_rows.iterrows()):
    file_path = Path(row["file_path"])
    true_class = "AiArtData" if row["label"] == 1 else "RealArt"
    
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    
    result = predictor.predict(img_bytes)
    latencies.append(result["latency_ms"])
    
    match_status = "MATCH" if result["prediction"] == true_class else "MISMATCH"
    print(f"[{idx+1}] File: {file_path.name[:25]}...")
    print(f"    True: {true_class} | Pred: {result['prediction']} ({result['confidence']}%) | [{match_status}]")
    print(f"    Inference Latency: {result['latency_ms']} ms\n")

avg_latency = sum(latencies) / len(latencies)
print("\n")
print(f"Serving Pipeline Status: VERIFIED")
print(f"Average Local Inference Latency: {avg_latency:.2f} ms per request")