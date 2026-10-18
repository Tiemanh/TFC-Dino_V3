"""Prepare a reviewable Kaggle upload folder; never uploads or publishes data."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def package(raw_dir, prepared_dir, output, dataset_id, title, license_name):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", dataset_id):
        raise ValueError("Dataset ID must be owner/dataset-slug")
    metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    if len(metadata.get("class_names", [])) != 8:
        raise ValueError("Expected a prepared private dataset with exactly 8 classes")
    raw_files = sorted(raw_dir.rglob("*.csv"))
    if not raw_files:
        raise ValueError("No private recordings found. InGesture must not be relabelled as private data")
    # Validate everything before creating the output folder.
    for file in raw_files:
        provenance = json.loads(file.with_suffix(".json").read_text(encoding="utf-8"))
        if provenance.get("source_type") != "user_recorded":
            raise ValueError(f"Missing self-collection provenance: {file}")
    for name in ("train.pt", "test.pt"):
        if not (prepared_dir / name).is_file():
            raise FileNotFoundError(prepared_dir / name)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for file in raw_files:
        relative = file.relative_to(raw_dir)
        for source in (file, file.with_suffix(".json")):
            destination = output / "raw" / relative.parent / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest.append({"file": "raw/" + relative.as_posix(),
                         "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    for name in ("train.pt", "test.pt", "val.pt", "metadata.json"):
        if (prepared_dir / name).exists():
            (output / "processed").mkdir(exist_ok=True)
            shutil.copy2(prepared_dir / name, output / "processed" / name)
    kaggle_metadata = {"title": title, "id": dataset_id,
                       "licenses": [{"name": license_name}]}
    (output / "dataset-metadata.json").write_text(json.dumps(kaggle_metadata, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Smartphone gesture recordings\n\n"
        "User-recorded data. Participant IDs are pseudonyms. Each raw CSV contains one labelled take.\n"
        "Pipeline: timestamp sorting, finite-value cleaning, interpolation/resampling, recording split, "
        "train-only clipping/normalization, segmentation. See processed/metadata.json for labels and splits.\n"
        "Check consent, content, label definitions and the chosen license before publishing.\n",
        encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("datasets/Gesture"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/kaggle_private_upload"))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--license", required=True, dest="license_name")
    args = parser.parse_args()
    print(package(args.raw_dir, args.prepared_dir, args.output, args.dataset_id, args.title, args.license_name))


if __name__ == "__main__":
    main()
