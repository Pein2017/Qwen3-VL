#!/usr/bin/env python3
"""
Teacher pool selection for data conversion pipeline.

Extracted from unified_processor.TeacherSelector with no behavior changes.
"""

import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from data_conversion.utils.file_ops import FileOperations


logger = logging.getLogger(__name__)


class TeacherSelector:
    """Deterministic teacher pool builder using fixed vocabulary coverage.

    Implements a greedy set-cover over a fixed universe of canonical tokens
    derived from attribute taxonomy/mapping. Falls back to a simple free
    vocabulary if the taxonomy/mapping files are unavailable.
    """

    FREE_TOP_K_PER_TYPE: int = 50

    # Negative/issue-oriented tokens to emphasize during selection (coverage-focused)
    NEGATIVE_TOKENS: Set[str] = {
        "未拧紧",
        "露铜",
        "复接",
        "生锈",
        "安装方向错误",
        "无保护措施",
        "弯曲半径不合理（弯曲半径<4cm或者成环）",
    }

    def __init__(
        self,
        label_hierarchy: Dict[str, List[str]],
        allowed_object_types: Set[str],
        max_teachers: int = 10,
        seed: int = 42,
    ):
        self.label_hierarchy = label_hierarchy
        self.allowed_object_types = allowed_object_types
        self.max_teachers = max_teachers
        self.seed = seed

        # Chinese → English object type mapping (static fallback)
        self.cn2en_types: Dict[str, str] = {
            "BBU设备": "bbu",
            "挡风板": "bbu_shield",
            "螺丝、光纤插头": "connect_point",
            "标签": "label",
            "光纤": "fiber",
            "电线": "wire",
        }

        # Load fixed universe from taxonomy/mapping if available
        (
            self.mode,
            self.universe_units,
            self.unit_to_objtypes,
            self.objtype_cn_by_en,
        ) = self._build_universe()

        logger.info(
            f"Initialized TeacherSelector mode={self.mode}, units={len(self.universe_units)}, max_teachers={max_teachers}"
        )

    def _build_universe(
        self,
    ) -> Tuple[str, Set[str], Dict[str, Set[str]], Dict[str, str]]:
        """Build coverage universe from fixed taxonomy/mapping; fallback to free mode.

        Returns:
            (mode, universe_units, unit_to_objtypes, objtype_cn_by_en)
        """
        base_dir = Path(__file__).parent
        taxonomy_path = base_dir / "attribute_taxonomy.json"
        mapping_path = base_dir / "hierarchical_attribute_mapping.json"

        if taxonomy_path.exists() and mapping_path.exists():
            try:
                taxonomy = FileOperations.load_json_data(taxonomy_path)
                mapping = FileOperations.load_json_data(mapping_path)
                units: Set[str] = set()
                unit_to_objtypes: Dict[str, Set[str]] = defaultdict(set)
                objtype_cn_by_en: Dict[str, str] = {}

                # From mapping: object types and attributes
                obj_types_map = mapping.get("object_types", {})
                for en_type, spec in obj_types_map.items():
                    cn_label = spec.get("chinese_label", "").strip()
                    if cn_label:
                        units.add(cn_label)
                        unit_to_objtypes[cn_label].add(en_type)
                        objtype_cn_by_en[en_type] = cn_label

                    attributes = spec.get("attributes", [])
                    for attr in attributes:
                        if attr.get("is_free_text"):
                            continue  # exclude free text
                        values = attr.get("values")
                        if isinstance(values, dict):
                            kv_pairs = list(values.items())
                        elif isinstance(values, list):
                            kv_pairs = [(v, v) for v in values]
                        else:
                            kv_pairs = []

                        for k, v in kv_pairs:
                            for token in self._split_tokens(k) | self._split_tokens(v):
                                units.add(token)
                                unit_to_objtypes[token].add(en_type)

                # Also incorporate explicit values from taxonomy attribute_groups
                attr_groups = taxonomy.get("attribute_groups", {})
                for group in attr_groups.values():
                    attrs = group.get("attributes", {})
                    for attr in attrs.values():
                        vals = attr.get("values")
                        if vals == "free_text":
                            continue
                        if isinstance(vals, dict):
                            candidates = list(vals.keys()) + list(vals.values())
                        elif isinstance(vals, list):
                            candidates = vals
                        else:
                            candidates = []
                        for c in candidates:
                            for token in self._split_tokens(c):
                                units.add(token)
                                # Heuristic association: if applies_to present, map tokens to those types
                                applies = attr.get("applies_to")
                                if isinstance(applies, list):
                                    for en_type in applies:
                                        unit_to_objtypes[token].add(en_type)

                # Filter units by allowed object types if mappings are exclusive
                if self.allowed_object_types:
                    filtered_units: Set[str] = set()
                    for token in units:
                        types = unit_to_objtypes.get(token)
                        if not types:
                            filtered_units.add(token)
                        elif set(types) & self.allowed_object_types:
                            filtered_units.add(token)
                    units = filtered_units

                # Remove occlusion-related tokens from the universe entirely
                if units:
                    units = {t for t in units if ("遮挡" not in t)}
                    # Also prune unit_to_objtypes for removed tokens
                    for t in list(unit_to_objtypes.keys()):
                        if "遮挡" in t:
                            unit_to_objtypes.pop(t, None)

                return "fixed", units, unit_to_objtypes, objtype_cn_by_en
            except Exception as e:
                logger.warning(
                    f"Failed loading fixed taxonomy/mapping: {e}; falling back to free mode"
                )

        # Fallback: universe will be derived later from dataset (free mode)
        from collections import defaultdict as _dd

        return "free", set(), _dd(set), {}

    @staticmethod
    def _split_tokens(text: str) -> Set[str]:
        if not isinstance(text, str) or not text.strip():
            return set()
        parts: List[str] = []
        for level in text.split("/"):
            parts.extend([p.strip() for p in level.split(",")])
        return {p for p in parts if p}

    @staticmethod
    def _geometry_types(sample: Dict) -> Set[str]:
        g: Set[str] = set()
        for obj in sample.get("objects", []):
            if "bbox_2d" in obj:
                g.add("bbox_2d")
            if "quad" in obj:
                g.add("quad")
            if "line" in obj:
                g.add("line")
        return g

    def _extract_units(self, sample: Dict) -> Set[str]:
        tokens: Set[str] = set()
        for obj in sample.get("objects", []):
            desc = obj.get("desc", "")
            if not desc:
                continue
            tokens |= self._split_tokens(desc)
        # Ignore occlusion-related tokens (contain '遮挡') as they add low training value
        if tokens:
            tokens = {t for t in tokens if ("遮挡" not in t)}
        if self.mode == "fixed":
            return tokens & self.universe_units
        return tokens

    def _detect_brand(self, sample_tokens: Set[str]) -> str:
        for brand in ("华为", "中兴", "爱立信"):
            if brand in sample_tokens:
                return brand
        return "unknown"

    def _build_free_universe(self, samples: List[Dict]) -> Set[str]:
        # Build per-type token DF and select top-K per allowed type
        df_by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in samples:
            seen_in_sample_by_type: Dict[str, Set[str]] = defaultdict(set)
            for obj in s.get("objects", []):
                desc = obj.get("desc", "")
                if not desc:
                    continue
                parts = [p.strip() for p in desc.split("/") if p.strip()]
                obj_type_cn = parts[0] if parts else ""
                en_type = self.cn2en_types.get(obj_type_cn, "unknown")
                if en_type != "unknown" and en_type not in self.allowed_object_types:
                    continue
                tokens = self._split_tokens(desc)
                seen_in_sample_by_type[en_type] |= tokens
            for en_type, toks in seen_in_sample_by_type.items():
                for t in toks:
                    df_by_type[en_type][t] += 1

        universe: Set[str] = set()
        for en_type, counter in df_by_type.items():
            if en_type == "unknown":
                continue
            # Select top-K tokens
            top = sorted(counter.items(), key=lambda kv: -kv[1])[: self.FREE_TOP_K_PER_TYPE]
            for token, _ in top:
                if "遮挡" in token:
                    continue  # ignore occlusion tokens entirely
                universe.add(token)
                self.unit_to_objtypes[token].add(en_type)
        return universe

    def _compute_idf_weights(self, per_sample_units: List[Set[str]]) -> Dict[str, float]:
        """Compute IDF-like weights over the current universe to emphasize rare tokens."""
        N = len(per_sample_units)
        df: Dict[str, int] = defaultdict(int)
        for units in per_sample_units:
            for u in units:
                df[u] += 1
        idf: Dict[str, float] = {}
        for u in self.universe_units:
            d = df.get(u, 0)
            idf[u] = math.log(1.0 + (N / (1.0 + d))) if N > 0 else 0.0
        return idf

    @staticmethod
    def _polyline_length(points: List[int]) -> float:
        if not points or len(points) < 4:
            return 0.0
        total = 0.0
        for i in range(0, len(points) - 2, 2):
            x1, y1 = points[i], points[i + 1]
            x2, y2 = points[i + 2], points[i + 3]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            total += math.hypot(dx, dy)
        return total

    def _compute_sample_meta(self, sample: Dict) -> Dict[str, float | int | bool | List[str]]:
        width = int(sample.get("width", 0) or 0)
        height = int(sample.get("height", 0) or 0)
        img_area = float(width * height) if width > 0 and height > 0 else 0.0
        diag = math.hypot(float(width), float(height)) if width > 0 and height > 0 else 1.0

        small_box_count = 0
        total_box_count = 0
        line_points = 0
        line_length = 0.0
        negative_hits: Set[str] = set()

        for obj in sample.get("objects", []) or []:
            desc = obj.get("desc", "") or ""
            toks = self._split_tokens(desc)
            for t in toks:
                if t in self.NEGATIVE_TOKENS:
                    negative_hits.add(t)

            if "bbox_2d" in obj:
                total_box_count += 1
                if img_area > 0:
                    x1, y1, x2, y2 = obj["bbox_2d"]
                    area = float(max(0, x2 - x1) * max(0, y2 - y1))
                    if area / img_area < 0.003:  # <0.3% of image area
                        small_box_count += 1
            if "line" in obj:
                pts = obj["line"]
                line_points += max(0, len(pts) // 2)
                line_length += self._polyline_length(pts)

        small_box_frac = (float(small_box_count) / float(total_box_count)) if total_box_count > 0 else 0.0
        line_length_norm = (line_length / diag) if diag > 0 else 0.0
        has_negative = len(negative_hits) > 0

        # Simple difficulty heuristic
        if has_negative or line_length_norm > 0.6 or small_box_frac > 0.2:
            difficulty = "hard"
        elif line_length_norm < 0.2 and small_box_frac < 0.05 and not has_negative:
            difficulty = "easy"
        else:
            difficulty = "medium"

        return {
            "small_box_frac": small_box_frac,
            "line_points": int(line_points),
            "line_length_norm": float(line_length_norm),
            "has_negative": has_negative,
            "negatives": sorted(list(negative_hits)),
            "difficulty": difficulty,
        }

    def select_teachers(self, samples: List[Dict]) -> Tuple[List[Dict], List[int], Dict]:
        """Deterministic greedy selection based on fixed rules.

        Returns:
            teacher_samples, selected_indices, stats_dict
        """
        if not samples:
            logger.warning("No samples provided for teacher selection")
            return [], [], {}

        # Handle explicit zero teachers (dynamic teacher-sampling)
        if self.max_teachers == 0:
            logger.info("max_teachers=0: Skipping teacher selection for dynamic teacher-sampling")
            stats = {
                "mode": self.mode,
                "pool_size": 0,
                "max_teachers": self.max_teachers,
            }
            return [], [], stats

        if len(samples) <= self.max_teachers:
            logger.info(
                f"Using all {len(samples)} samples as teachers (below max_teachers)"
            )
            # Minimal stats
            stats = {
                "mode": self.mode,
                "pool_size": len(samples),
                "max_teachers": self.max_teachers,
            }
            return samples, list(range(len(samples))), stats

        # Build universe if in free mode
        if self.mode == "free":
            self.universe_units = self._build_free_universe(samples)

        # Precompute per-sample metadata
        per_sample_units: List[Set[str]] = []
        per_sample_tokens: List[Set[str]] = []
        per_sample_geometry: List[Set[str]] = []
        per_sample_brand: List[str] = []
        object_counts: List[int] = []
        per_sample_meta: List[Dict] = []
        for s in samples:
            tokens = self._extract_units(s) if self.mode == "fixed" else self._extract_units(s)
            per_sample_tokens.append(tokens)
            per_sample_units.append(tokens if self.mode == "fixed" else (tokens & self.universe_units))
            per_sample_geometry.append(self._geometry_types(s))
            per_sample_brand.append(self._detect_brand(tokens))
            object_counts.append(len(s.get("objects", [])))
            per_sample_meta.append(self._compute_sample_meta(s))

        median_objects = statistics.median(object_counts) if object_counts else 0
        # Dataset-level brand distribution and line presence
        brand_df: Dict[str, int] = defaultdict(int)
        for b in per_sample_brand:
            if b != "unknown":
                brand_df[b] += 1
        total_known_brands = sum(brand_df.values()) or 1
        brand_target: Dict[str, int] = {}
        for b, cnt in brand_df.items():
            share = float(cnt) / float(total_known_brands)
            brand_target[b] = max(0, int(round(self.max_teachers * share)))
        # Ensure at least 1 for any present brand if capacity allows
        leftover = self.max_teachers - sum(brand_target.values())
        if leftover > 0:
            # Distribute leftover to brands with highest dataset share
            for b, _ in sorted(brand_df.items(), key=lambda kv: -kv[1]):
                if leftover <= 0:
                    break
                brand_target[b] += 1
                leftover -= 1

        line_dataset_ratio = (
            sum(1 for g in per_sample_geometry if "line" in g) / float(len(per_sample_geometry))
            if per_sample_geometry
            else 0.0
        )
        # Encourage lines at least to dataset proportion, with a floor
        target_line_ratio = max(0.30, line_dataset_ratio)

        selected: List[int] = []
        units_remaining: Set[str] = set(self.universe_units)
        selected_geometries: Set[str] = set()
        brand_counts: Dict[str, int] = defaultdict(int)
        selected_line_count: int = 0

        # IDF weights over universe for coverage-driven rare token emphasis
        idf_weights = self._compute_idf_weights(per_sample_units)

        # Helper to produce sort key per candidate (recomputed each round)
        def candidate_key(idx: int) -> Tuple:
            # IDF-weighted units hit
            hit_units = per_sample_units[idx] & units_remaining
            units_hit = sum(idf_weights.get(u, 0.0) for u in hit_units)
            geom = per_sample_geometry[idx]
            has_line = 1 if "line" in geom else 0
            # Only activate line preference if fiber/wire-related units remain
            fiber_wire_remaining = any(
                (
                    self.unit_to_objtypes.get(u) and (self.unit_to_objtypes[u] & {"fiber", "wire"})
                )
                for u in units_remaining
            )
            # Line encouragement if under target ratio
            current_line_ratio = (selected_line_count / float(max(1, len(selected)))) if selected else 0.0
            line_needed = current_line_ratio < target_line_ratio
            line_pref = 0
            if fiber_wire_remaining and has_line:
                line_pref += 1
            if line_needed and has_line:
                line_pref += 1
            brand = per_sample_brand[idx]
            # Prefer brands under their dataset-proportional target
            on_target = brand_counts[brand] >= brand_target.get(brand, 0)
            brand_balance = brand_counts[brand] if on_target else 0
            geom_novelty = len(geom - selected_geometries)
            obj_delta = abs(object_counts[idx] - median_objects)
            # Lexicographic fallback by first image path
            img_path = ""
            imgs = samples[idx].get("images") or []
            if imgs:
                img_path = str(imgs[0])
            # Sort by: more units_hit (IDF), more line_pref, better brand balance, more geom novelty, closer to median count, lexicographic
            return (-units_hit, -line_pref, brand_balance, -geom_novelty, obj_delta, img_path)

        # Phase B: Greedy cover until cap or universe exhausted
        candidate_indices = list(range(len(samples)))
        while units_remaining and len(selected) < self.max_teachers:
            # Filter candidates that contribute anything
            contributing = [i for i in candidate_indices if (per_sample_units[i] & units_remaining)]
            if not contributing:
                break
            best = min(contributing, key=candidate_key)
            selected.append(best)
            units_remaining -= per_sample_units[best]
            selected_geometries |= per_sample_geometry[best]
            brand_counts[per_sample_brand[best]] += 1
            if "line" in per_sample_geometry[best]:
                selected_line_count += 1
            candidate_indices.remove(best)

        # Phase C: Cap and fill for diversity if capacity remains
        if len(selected) < self.max_teachers and candidate_indices:

            def fill_key(idx: int) -> Tuple:
                geom = per_sample_geometry[idx]
                geom_novelty = len(geom - selected_geometries)
                brand = per_sample_brand[idx]
                brand_balance = brand_counts[brand]
                pool_units_seen = self.universe_units - units_remaining
                pool_novelty = len(per_sample_units[idx] - pool_units_seen)
                obj_delta = abs(object_counts[idx] - median_objects)
                img_path = ""
                imgs = samples[idx].get("images") or []
                if imgs:
                    img_path = str(imgs[0])
                # Prefer higher geom_novelty, lower brand_balance, higher pool_novelty, lower obj_delta
                return (-geom_novelty, brand_balance, -pool_novelty, obj_delta, img_path)

            remaining_sorted = sorted(candidate_indices, key=fill_key)
            for idx in remaining_sorted:
                if len(selected) >= self.max_teachers:
                    break
                selected.append(idx)
                selected_geometries |= per_sample_geometry[idx]
                brand_counts[per_sample_brand[idx]] += 1
                if "line" in per_sample_geometry[idx]:
                    selected_line_count += 1

        selected = sorted(set(selected))
        teacher_samples: List[Dict] = []
        # Attach lightweight meta info for downstream pairing (optional)
        for i in selected:
            sample = samples[i].copy()
            tokens_list = sorted(list(per_sample_tokens[i]))
            meta_extra = per_sample_meta[i]
            sample["meta"] = {
                "tokens": tokens_list,
                "object_types_present": sorted(list({
                    self.cn2en_types.get(next(iter(self._split_tokens(obj.get("desc", "").split("/")[0]) or []), ""), "unknown")
                    for obj in sample.get("objects", []) or []
                } - {"unknown"})),
                "geometry_set": sorted(list(per_sample_geometry[i])),
                "brand": per_sample_brand[i],
                "object_count": object_counts[i],
                **meta_extra,
            }
            teacher_samples.append(sample)

        # Stats
        covered_units = sorted(list(self.universe_units - units_remaining))
        total_idf = sum(idf_weights.get(u, 0.0) for u in self.universe_units) or 1.0
        covered_idf = sum(idf_weights.get(u, 0.0) for u in covered_units)
        # Negative token coverage
        neg_df_dataset: Dict[str, int] = defaultdict(int)
        for toks in per_sample_tokens:
            for t in toks:
                if t in self.NEGATIVE_TOKENS:
                    neg_df_dataset[t] += 1
        neg_covered: Dict[str, int] = defaultdict(int)
        for i in selected:
            for t in per_sample_tokens[i]:
                if t in self.NEGATIVE_TOKENS:
                    neg_covered[t] += 1
        stats: Dict = {
            "mode": self.mode,
            "pool_size": len(selected),
            "max_teachers": self.max_teachers,
            "universe_size": len(self.universe_units),
            "covered_units_count": len(covered_units),
            "coverage_ratio": float(len(covered_units) / len(self.universe_units)) if self.universe_units else 0.0,
            "uncovered_units": sorted(list(units_remaining)) if units_remaining else [],
            "brand_distribution": dict(brand_counts),
            "geometry_presence": sorted(list(selected_geometries)),
            "idf_coverage_ratio": float(covered_idf / total_idf) if total_idf > 0 else 0.0,
            "brand_distribution_dataset": dict(brand_df),
            "line_ratio_dataset": line_dataset_ratio,
            "line_ratio_selected": (selected_line_count / float(max(1, len(selected)))) if selected else 0.0,
            "negative_tokens_dataset": dict(neg_df_dataset),
            "negative_tokens_selected": dict(neg_covered),
        }
        pool_object_counts = [object_counts[i] for i in selected]
        if pool_object_counts:
            sorted_counts = sorted(pool_object_counts)
            p95_idx = max(0, min(len(sorted_counts) - 1, int(0.95 * (len(sorted_counts) - 1))))
            stats["object_count_summary"] = {
                "min": sorted_counts[0],
                "median": statistics.median(sorted_counts),
                "p95": p95_idx and sorted_counts[p95_idx] or sorted_counts[-1],
            }

        logger.info(
            f"Teacher pool built: size={len(selected)} coverage={stats.get('covered_units_count')}/{stats.get('universe_size')}"
        )

        return teacher_samples, selected, stats
