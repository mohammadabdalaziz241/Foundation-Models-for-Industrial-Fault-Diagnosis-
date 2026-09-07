"""Inventory official CWRU sources without fitting or analysing signal values.

Downloads only the official 12k DE/FE categories when --download is supplied.
The 48k category is recorded as a denylist; its MAT files are never downloaded.
MAT headers and SHA-256 hashes establish source inventory, not a frozen fold
registry. Current Otter identities/splits and window settings are still needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request
from urllib.parse import urljoin, urlparse

BASE = "https://engineering.case.edu/bearingdatacenter/"
PAGES = {
    "official_12k_DE_fault": ("12k-drive-end-bearing-fault-data", "DE", 12000),
    "official_12k_FE_fault": ("12k-fan-end-bearing-fault-data", "FE", 12000),
    "official_48k_DE_fault": ("48k-drive-end-bearing-fault-data", "DE", 48000),
}
LABEL = re.compile(r"^(IR|B|OR)(\d{3})(?:@(\d+))?_(\d)$")
CHANNEL = re.compile(r"^(?:X\d+_)?(DE|FE|BA)_time$")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.parts = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.parts = []

    def handle_data(self, data):
        if self.href is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "a":
            if self.href:
                self.links.append(("".join(self.parts).strip(), self.href))
            self.href = None
            self.parts = []


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "PCSTE-native12-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        if urlparse(response.geturl()).hostname != "engineering.case.edu":
            raise ValueError("Source redirected outside the official CWRU host")
        data = response.read()
    tmp = destination.with_suffix(destination.suffix + ".partial")
    tmp.write_bytes(data)
    tmp.replace(destination)
    return data


def catalogue(category, work):
    slug, fault_end, rate = PAGES[category]
    page_url = BASE + slug
    raw = fetch(page_url, work / "source_pages" / (category + ".html"))
    parser = Links()
    parser.feed(raw.decode("utf-8"))
    rows = []
    for label, href in parser.links:
        match = LABEL.fullmatch(label)
        if match is None:
            continue
        url = urljoin(page_url, href)
        file_match = re.fullmatch(r"/sites/default/files/(\d+)\.mat", urlparse(url).path)
        if urlparse(url).hostname != "engineering.case.edu" or file_match is None:
            raise ValueError(f"Unexpected official file link: {label}: {url}")
        kind, diameter, orientation, load = match.groups()
        number = int(file_match.group(1))
        rows.append({
            "official_file_number": number,
            "official_filename": f"{number}.mat",
            "official_label": label,
            "source_category": category,
            "source_page": page_url,
            "source_url": url,
            "native_sampling_rate_hz": rate,
            "fault_bearing_end": fault_end,
            "fault_class": {"IR": "InnerRace", "B": "Ball", "OR": "OuterRace"}[kind],
            "fault_diameter_mil": int(diameter),
            "outer_race_orientation_clock": int(orientation) if orientation else None,
            "load_hp": int(load),
            "conservative_identity_candidate": f"{fault_end}_{kind}{diameter}",
            "identity_status": "PENDING_CURRENT_PHYSICAL_REGISTRY_RECONCILIATION",
            "eligibility": "NATIVE12_SOURCE_CANDIDATE" if rate == 12000 else "EXCLUDED_48K_SOURCE",
        })
    if not rows or len({r["official_file_number"] for r in rows}) != len(rows):
        raise ValueError(f"Empty/duplicate official catalogue: {category}")
    return rows, {"url": page_url, "sha256": sha256(raw).hexdigest(), "catalogue_entries": len(rows)}


def inspect_native(row, work):
    if row["native_sampling_rate_hz"] != 12000 or row["source_category"] not in (
        "official_12k_DE_fault", "official_12k_FE_fault"
    ):
        raise ValueError("Refusing to download a source outside the native12 allowlist")
    result = dict(row)
    path = work / "raw_native12" / row["source_category"] / row["official_filename"]
    try:
        from scipy.io import whosmat
        data = fetch(row["source_url"], path)
        headers = whosmat(path)
        channels = []
        for name, shape, dtype in headers:
            match = CHANNEL.fullmatch(name)
            if match:
                n = 1
                for size in shape:
                    n *= size
                channels.append({"variable": name, "sensor_location": match.group(1),
                                 "shape": list(shape), "sample_count": n, "matlab_class": dtype})
        if not any(c["sensor_location"] in ("DE", "FE") for c in channels):
            raise ValueError("No recognised DE/FE signal channel in MAT headers")
        result.update(file_sha256=sha256(data).hexdigest(), file_size_bytes=len(data),
                      local_path=str(path.resolve()), channel_headers=channels,
                      mat_variables=[name for name, _, _ in headers],
                      source_verification="OFFICIAL_NATIVE12_DOWNLOAD_AND_MAT_HEADERS_VERIFIED")
    except Exception as exc:
        result.update(source_verification="SOURCE_AUDIT_BLOCKED", reason=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    rows, pages = [], {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(catalogue, category, work): category for category in PAGES}
        for future in as_completed(futures):
            category = futures[future]
            entries, page = future.result()
            rows.extend(entries)
            pages[category] = page
    native = sorted((r for r in rows if r["native_sampling_rate_hz"] == 12000),
                    key=lambda r: (r["fault_bearing_end"], r["official_file_number"]))
    excluded = sorted((r for r in rows if r["native_sampling_rate_hz"] == 48000),
                      key=lambda r: r["official_file_number"])
    if {r["official_file_number"] for r in native} & {r["official_file_number"] for r in excluded}:
        raise ValueError("An acquisition ID is listed in both 12k and 48k categories")
    if args.download:
        completed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(inspect_native, row, work) for row in native]
            for future in as_completed(futures):
                completed.append(future.result())
                if len(completed) % 10 == 0 or len(completed) == len(native):
                    print(json.dumps({"inspected": len(completed), "total": len(native),
                                      "blocked": sum(r["source_verification"] == "SOURCE_AUDIT_BLOCKED"
                                                     for r in completed)}), flush=True)
        native = sorted(completed, key=lambda r: (r["fault_bearing_end"], r["official_file_number"]))
    groups = defaultdict(set)
    for row in native:
        groups[row["fault_class"]].add(row["conservative_identity_candidate"])
    hashes = defaultdict(list)
    for row in native:
        if row.get("file_sha256"):
            hashes[row["file_sha256"]].append(row["official_file_number"])
    output = {
        "schema_version": "cwru_native12_source_inventory_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": "SOURCE_INVENTORY_ONLY_NOT_FROZEN_NO_TRAINING",
        "source_pages": pages,
        "native12_catalogue_count": len(native),
        "verified_native12_download_count": sum(r.get("source_verification", "").endswith("HEADERS_VERIFIED") for r in native),
        "catalogue_counts_by_fault_end_and_class": dict(Counter(r["fault_bearing_end"] + ":" + r["fault_class"] for r in native)),
        "conservative_identity_candidates_by_class": {c: sorted(v) for c, v in sorted(groups.items())},
        "identity_warning": "These are conservative end/class/diameter candidates, not independently established physical specimens. Reconcile with the current Otter identity and sealed role registry.",
        "native12_files": native,
        "excluded_official_48k_fault_files": excluded,
        "additional_exclusion_rule": "Exclude official normal baseline 97-100.mat and aliases: the repository's frozen acquisition-rate audit identifies them as 48k even in a 12k-named local directory. No Healthy class is added.",
        "duplicate_download_hashes": {h: ids for h, ids in hashes.items() if len(ids) > 1},
        "missing_for_protocol_freeze": ["Current Otter physical identity and per-fold sealed-role registries", "Current window/STFT parameters and provenance", "Current unchanged JNU/HIT/MaFaulDa manifest and normaliser hashes", "Native12 CWRU TRAIN-only normalisers", "Complete ten-item audit and versioned protocol digest"],
    }
    destination = work / "OFFICIAL_NATIVE12_SOURCE_INVENTORY.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"inventory": str(destination), "native12_catalogue_count": len(native),
                      "verified": output["verified_native12_download_count"],
                      "excluded_48k_catalogue_count": len(excluded),
                      "protocol_status": output["protocol_status"]}), flush=True)


if __name__ == "__main__":
    main()
