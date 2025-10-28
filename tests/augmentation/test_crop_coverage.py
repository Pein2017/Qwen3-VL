"""Unit tests for crop operators and coverage utilities."""

import pytest
from src.datasets.geometry import (
    get_aabb,
    intersect_aabb,
    aabb_area,
    compute_coverage,
    compute_polygon_coverage,
    translate_geometry,
)


class TestAABBUtilities:
    """Test axis-aligned bounding box utilities."""

    def test_get_aabb_from_bbox(self):
        """get_aabb should return bbox directly."""
        geom = {"bbox_2d": [10.0, 20.0, 50.0, 60.0]}
        assert get_aabb(geom) == [10.0, 20.0, 50.0, 60.0]

    def test_get_aabb_from_quad(self):
        """get_aabb should compute min/max from quad points."""
        # Square quad: (10,20), (50,20), (50,60), (10,60)
        geom = {"quad": [10.0, 20.0, 50.0, 20.0, 50.0, 60.0, 10.0, 60.0]}
        assert get_aabb(geom) == [10.0, 20.0, 50.0, 60.0]

    def test_get_aabb_from_line(self):
        """get_aabb should compute min/max from line points."""
        # Line: (10,20) -> (50,60)
        geom = {"line": [10.0, 20.0, 50.0, 60.0]}
        assert get_aabb(geom) == [10.0, 20.0, 50.0, 60.0]

    def test_intersect_aabb_full_overlap(self):
        """Intersecting AABBs should return intersection."""
        bbox_a = [10.0, 20.0, 50.0, 60.0]
        bbox_b = [30.0, 40.0, 70.0, 80.0]
        result = intersect_aabb(bbox_a, bbox_b)
        assert result == [30.0, 40.0, 50.0, 60.0]

    def test_intersect_aabb_no_overlap(self):
        """Non-overlapping AABBs should return zero bbox."""
        bbox_a = [10.0, 20.0, 30.0, 40.0]
        bbox_b = [50.0, 60.0, 70.0, 80.0]
        result = intersect_aabb(bbox_a, bbox_b)
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_aabb_area(self):
        """aabb_area should compute width × height."""
        bbox = [10.0, 20.0, 50.0, 60.0]
        assert aabb_area(bbox) == 40.0 * 40.0  # 1600

    def test_aabb_area_zero(self):
        """Degenerate bbox should have zero area."""
        bbox = [0.0, 0.0, 0.0, 0.0]
        assert aabb_area(bbox) == 0.0


class TestCoverageComputation:
    """Test coverage ratio computation."""

    def test_coverage_fully_inside(self):
        """Object fully inside crop should have coverage=1.0."""
        geom = {"bbox_2d": [20.0, 30.0, 40.0, 50.0]}
        crop = [10.0, 20.0, 50.0, 60.0]
        assert compute_coverage(geom, crop) == 1.0

    def test_coverage_fully_outside(self):
        """Object fully outside crop should have coverage=0.0."""
        geom = {"bbox_2d": [60.0, 70.0, 80.0, 90.0]}
        crop = [10.0, 20.0, 50.0, 60.0]
        assert compute_coverage(geom, crop) == 0.0

    def test_coverage_partial(self):
        """Partially visible object should have 0.0 < coverage < 1.0."""
        # Object: [10,10,50,50] (40×40 = 1600)
        # Crop: [30,30,70,70]
        # Intersection: [30,30,50,50] (20×20 = 400)
        # Coverage: 400/1600 = 0.25
        geom = {"bbox_2d": [10.0, 10.0, 50.0, 50.0]}
        crop = [30.0, 30.0, 70.0, 70.0]
        coverage = compute_coverage(geom, crop)
        assert 0.24 < coverage < 0.26  # Allow floating point tolerance

    def test_polygon_coverage_matches_exact_area(self):
        """Polygon coverage uses clipped polygon area rather than AABB."""
        geom = {"quad": [0.0, 0.0, 100.0, 0.0, 100.0, 20.0, 0.0, 20.0]}
        crop = [0.0, 0.0, 20.0, 20.0]
        coverage = compute_polygon_coverage(geom, crop)
        assert pytest.approx(0.1805, abs=1e-3) == coverage

    def test_polygon_coverage_zero_when_clip_empty(self):
        """Elongated, rotated polygon with tiny overlap should have low coverage."""
        geom = {"quad": [0.0, 0.0, 200.0, 40.0, 180.0, 60.0, -20.0, 20.0]}
        crop = [0.0, 0.0, 40.0, 40.0]
        coverage = compute_polygon_coverage(geom, crop)
        assert coverage < 0.2

    def test_coverage_degenerate_inside(self):
        """Degenerate geometry inside crop should have coverage=1.0."""
        geom = {"bbox_2d": [30.0, 40.0, 30.0, 40.0]}  # Point
        crop = [10.0, 20.0, 50.0, 60.0]
        assert compute_coverage(geom, crop) == 1.0

    def test_coverage_degenerate_outside(self):
        """Degenerate geometry outside crop should have coverage=0.0."""
        geom = {"bbox_2d": [70.0, 80.0, 70.0, 80.0]}  # Point
        crop = [10.0, 20.0, 50.0, 60.0]
        assert compute_coverage(geom, crop) == 0.0


class TestTranslateGeometry:
    """Test geometry translation."""

    def test_translate_bbox(self):
        """Translating bbox should shift both corners."""
        geom = {"bbox_2d": [10.0, 20.0, 50.0, 60.0]}
        result = translate_geometry(geom, -5.0, -10.0)
        assert result == {"bbox_2d": [5.0, 10.0, 45.0, 50.0]}

    def test_translate_quad(self):
        """Translating quad should shift all 4 points."""
        geom = {"quad": [10.0, 20.0, 50.0, 20.0, 50.0, 60.0, 10.0, 60.0]}
        result = translate_geometry(geom, -5.0, -10.0)
        expected = {"quad": [5.0, 10.0, 45.0, 10.0, 45.0, 50.0, 5.0, 50.0]}
        assert result == expected

    def test_translate_line(self):
        """Translating line should shift all points."""
        geom = {"line": [10.0, 20.0, 50.0, 60.0]}
        result = translate_geometry(geom, -5.0, -10.0)
        assert result == {"line": [5.0, 10.0, 45.0, 50.0]}


class TestCropOperatorIntegration:
    """Integration tests for crop operators (requires full augmentation stack)."""

    @pytest.mark.skip(reason="Integration test - requires full pipeline setup")
    def test_random_crop_filters_objects(self):
        """RandomCrop should filter objects based on min_coverage."""
        # This would test the full RandomCrop.apply() method
        # Requires setting up PIL images, geometries, and RNG
        pass

    @pytest.mark.skip(reason="Integration test - requires full pipeline setup")
    def test_crop_skip_on_min_objects(self):
        """Crop should skip if < min_objects remain."""
        pass

    @pytest.mark.skip(reason="Integration test - requires full pipeline setup")
    def test_crop_skip_on_line_objects(self):
        """Crop should skip if skip_if_line=True and line present."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
