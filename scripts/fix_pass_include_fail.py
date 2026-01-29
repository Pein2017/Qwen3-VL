#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import io
from collections import OrderedDict, defaultdict


def load_jsonl(path):
    entries = []
    with io.open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s, object_pairs_hook=OrderedDict)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse JSON on line {ln} of {path}: {e}\nLine: {s[:200]}..."
                )
            entries.append(obj)
    return entries


def dump_jsonl(entries, path):
    with io.open(path, "w", encoding="utf-8") as f:
        for e in entries:
            # enforce key order
            ordered = OrderedDict()
            for k in ("group_id", "mission", "label", "images", "per_image"):
                if k in e:
                    ordered[k] = e[k]
                else:
                    # keep any extra keys at the end in case they exist
                    pass
            # append any other keys while preserving their original order
            for k, v in e.items():
                if k not in ordered:
                    ordered[k] = v
            f.write(
                json.dumps(ordered, ensure_ascii=False, separators=(", ", ": ")) + "\n"
            )


def build_group_maps(entries):
    groups = defaultdict(lambda: {"pass": [], "fail": []})
    for e in entries:
        label = e.get("label")
        gid = e.get("group_id")
        if gid is None or label not in ("pass", "fail"):
            continue
        groups[gid][label].append(e)
    return groups


def fail_image_map(fail_entries):
    # Map image filename -> description from per_image
    m = OrderedDict()
    for fe in fail_entries:
        imgs = fe.get("images") or []
        per_img = fe.get("per_image") or {}
        for idx, img in enumerate(imgs, start=1):
            key = f"image_{idx}"
            desc = per_img.get(key, "")
            if img not in m:
                m[img] = desc
    return m


def ensure_pass_contains_fail(entries):
    groups = build_group_maps(entries)
    groups_with_both = [gid for gid, g in groups.items() if g["fail"] and g["pass"]]

    updated_count = 0
    added_images_total = 0

    for gid in groups_with_both:
        fmap = fail_image_map(groups[gid]["fail"])
        if not fmap:
            continue
        # Update every pass entry in this group
        for pe in groups[gid]["pass"]:
            imgs = list(pe.get("images") or [])
            per_img = OrderedDict(
                (k, v) for k, v in (pe.get("per_image") or {}).items()
            )

            existing = set(imgs)
            to_add = [(img, desc) for img, desc in fmap.items() if img not in existing]
            if not to_add:
                # already contains all fail images
                continue

            start_len = len(imgs)
            for i, (img, desc) in enumerate(to_add, start=1):
                imgs.append(img)
                per_img[f"image_{start_len + i}"] = desc

            pe["images"] = imgs
            pe["per_image"] = per_img
            updated_count += 1
            added_images_total += len(to_add)

    return groups_with_both, updated_count, added_images_total


def process(in_path, out_path):
    entries = load_jsonl(in_path)
    groups_with_both, updated_count, added_images_total = ensure_pass_contains_fail(
        entries
    )
    dump_jsonl(entries, out_path)
    print(
        json.dumps(
            {
                "file": in_path,
                "output": out_path,
                "groups_with_both": len(groups_with_both),
                "pass_entries_updated": updated_count,
                "images_added": added_images_total,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: fix_pass_include_fail.py <input_jsonl> <output_jsonl>")
        sys.exit(2)
    process(sys.argv[1], sys.argv[2])
