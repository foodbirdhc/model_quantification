#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def collect_json_files(root: Path) -> Dict[str, Path]:
    """Return {layer_name: file_path} for all .json files under root."""
    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")

    files = {}
    for p in sorted(root.rglob("*.json")):
        if p.is_file():
            # use relative path (without extension) as canonical name so nested dirs are preserved
            rel = p.relative_to(root).with_suffix("")
            files[str(rel)] = p
    return files


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_numeric_equal(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        return abs(float(a) - float(b)) <= (atol + rtol * max(abs(float(a)), abs(float(b))))
    return a == b


def compare_json_values(a: Any, b: Any, key_path: str = "root", rtol: float = 1e-5, atol: float = 1e-8):
    """Return a list of mismatches, or [] if equal."""
    mismatches = []

    if isinstance(a, dict) and isinstance(b, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        if a_keys != b_keys:
            mismatches.append(f"{key_path}: key set mismatch -> left={sorted(a_keys)}, right={sorted(b_keys)}")
            return mismatches

        for k in sorted(a_keys):
            mismatches.extend(compare_json_values(a[k], b[k], f"{key_path}.{k}", rtol=rtol, atol=atol))
        return mismatches

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            mismatches.append(f"{key_path}: list length mismatch -> left={len(a)}, right={len(b)}")
            return mismatches
        for i, (x, y) in enumerate(zip(a, b)):
            mismatches.extend(compare_json_values(x, y, f"{key_path}[{i}]", rtol=rtol, atol=atol))
        return mismatches

    if not check_numeric_equal(a, b, rtol=rtol, atol=atol):
        mismatches.append(f"{key_path}: left={a!r}, right={b!r}")

    return mismatches


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare layer output JSON files between two model exports.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./output/layer_outputs"),
        help="Root directory containing model folders, e.g. ./output/layer_outputs",
    )
    parser.add_argument("--left", default="onnx", help="Left model folder name, default: onnx")
    parser.add_argument("--right", default="torch", help="Right model folder name, default: torch")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for float comparison")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for float comparison")
    args = parser.parse_args()

    root = args.root.resolve()
    left_dir = root / args.left
    right_dir = root / args.right

    left_json_dir = left_dir / "json" if (left_dir / "json").exists() else left_dir
    right_json_dir = right_dir / "json" if (right_dir / "json").exists() else right_dir

    print_section("Model directories")
    print(f"left:  {left_json_dir}")
    print(f"right: {right_json_dir}")

    left_files = collect_json_files(left_json_dir)
    right_files = collect_json_files(right_json_dir)

    left_names = set(left_files.keys())
    right_names = set(right_files.keys())

    print_section("1) Layer count and names comparison")
    print(f"left layer count : {len(left_names)}")
    print(f"right layer count: {len(right_names)}")

    if left_names == right_names:
        print("Result: layer names are exactly consistent.")
    else:
        print("Result: layer names are NOT consistent.")
        missing_in_left = sorted(right_names - left_names)
        missing_in_right = sorted(left_names - right_names)
        if missing_in_left:
            print(f"Files only in right (missing in left): {len(missing_in_left)}")
            for name in missing_in_left[:20]:
                print(f"  - {name}")
        if missing_in_right:
            print(f"Files only in left (missing in right): {len(missing_in_right)}")
            for name in missing_in_right[:20]:
                print(f"  - {name}")

    common_names = sorted(left_names & right_names)
    print(f"common layer names: {len(common_names)}")
    # 2) Try direct JSON comparison for exact-name matches
    print_section("2) JSON value comparison for exact-name matches")
    mismatch_count = 0
    if common_names:
        for name in common_names:
            left_data = load_json(left_files[name])
            right_data = load_json(right_files[name])
            mismatches = compare_json_values(left_data, right_data, key_path=name, rtol=args.rtol, atol=args.atol)

            if mismatches:
                mismatch_count += 1
                print(f"[MISMATCH] {name}")
                for msg in mismatches[:10]:
                    print(f"  - {msg}")
                if len(mismatches) > 10:
                    print(f"  ... and {len(mismatches) - 10} more mismatches")

        if mismatch_count == 0:
            print("Result: all exact-name JSON values are consistent within tolerance.")
        else:
            print(f"Result: {mismatch_count} exact-name layer(s) have JSON value mismatches.")
    else:
        print("No exact-name common layers found. Will attempt best-effort matching by shape/value.")

    # 3) Best-effort alignment: match remaining left vs right by shape then by numeric closeness
    left_only = sorted(left_names - right_names)
    right_only = sorted(right_names - left_names)

    if not left_only or not right_only:
        print_section("3) Best-effort matching")
        print("No unmatched files to best-match.")
        return

    print_section("3) Best-effort matching by shape + numeric similarity")

    def json_shape(name, files_map, base_dir):
        try:
            d = load_json(files_map[name])
            return tuple(d.get("shape", []))
        except Exception:
            return None

    import numpy as np

    def npy_path_from_json(p: Path):
        # assume json is under .../json/<name>.json and npy is under .../npy/<name>.npy
        parts = p.parts
        # replace '/json/' with '/npy/' in path
        s = str(p)
        if "/json/" in s:
            return Path(s.replace("/json/", "/npy/").rsplit(".json", 1)[0] + ".npy")
        # fallback: sibling .npy
        return p.with_suffix("").with_name(p.stem).with_suffix('.npy')

    # build shape-index for right side
    right_shape_index = {}
    for name in right_only:
        shp = json_shape(name, right_files, right_json_dir)
        right_shape_index.setdefault(shp, []).append(name)

    matches = []
    unmatched_left = []

    for lname in left_only:
        lshp = json_shape(lname, left_files, left_json_dir)
        candidates = right_shape_index.get(lshp, [])
        if not candidates:
            unmatched_left.append(lname)
            continue

        # if single candidate, accept it; else pick by numeric closeness
        best = None
        best_score = None
        l_npy = npy_path_from_json(left_files[lname])
        if not l_npy.exists():
            unmatched_left.append(lname)
            continue
        l_arr = np.load(l_npy)

        for rname in candidates:
            r_npy = npy_path_from_json(right_files[rname])
            if not r_npy.exists():
                continue
            try:
                r_arr = np.load(r_npy)
            except Exception:
                continue

            # compute relative L2 error or max abs
            try:
                diff = np.linalg.norm(l_arr.ravel() - r_arr.ravel())
                denom = np.linalg.norm(l_arr.ravel())
                rel = diff / (denom + 1e-12)
            except Exception:
                rel = float('inf')

            if best is None or rel < best_score:
                best = rname
                best_score = rel

        if best is not None:
            matches.append((lname, best, best_score))
        else:
            unmatched_left.append(lname)

    # report matches
    print(f"Found {len(matches)} candidate matches based on shape + closeness.")
    for l, r, score in matches[:50]:
        print(f"  {l}  <--->  {r}    rel_error={score:.3e}")

    if unmatched_left:
        print(f"Unmatched left files: {len(unmatched_left)} (showing up to 20)")
        for n in unmatched_left[:20]:
            print(f"  - {n}")

    print("Best-effort matching completed. Review above matches and adjust thresholds if needed.")


if __name__ == "__main__":
    main()
