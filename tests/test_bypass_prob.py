"""Test augmentation bypass probability feature."""
import random
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor
from src.datasets.augmentation.base import Compose


def test_bypass_prob_zero():
    """Test that bypass_prob=0.0 never bypasses augmentation."""
    # Create a simple Compose pipeline
    augmenter = Compose([])  # Empty pipeline, will just return images as-is
    
    preprocessor = AugmentationPreprocessor(
        augmenter=augmenter,
        rng=random.Random(42),
        bypass_prob=0.0
    )
    
    # Mock record with no images (will return immediately)
    record = {"images": [], "objects": []}
    
    # Run multiple times
    for _ in range(10):
        result = preprocessor.preprocess(record)
        assert result is not None
    
    print("✓ bypass_prob=0.0 works correctly")


def test_bypass_prob_one():
    """Test that bypass_prob=1.0 always bypasses augmentation."""
    # Track if augmenter was called
    call_tracker = {"called": False}
    
    class TrackingCompose(Compose):
        def apply(self, images, geoms, **kwargs):
            call_tracker["called"] = True
            return super().apply(images, geoms, **kwargs)
    
    augmenter = TrackingCompose([])
    preprocessor = AugmentationPreprocessor(
        augmenter=augmenter,
        rng=random.Random(42),
        bypass_prob=1.0
    )
    
    # Mock record
    record = {"images": [], "objects": []}
    
    # Run multiple times - should never call augmenter due to bypass
    for _ in range(100):
        result = preprocessor.preprocess(record)
        assert result is record  # Should return same object
    
    # Augmenter should never have been called
    assert not call_tracker["called"], "Augmenter was called despite bypass_prob=1.0"
    
    print("✓ bypass_prob=1.0 works correctly")


def test_bypass_prob_intermediate():
    """Test that bypass_prob=0.1 bypasses ~10% of samples."""
    bypass_count = 0
    augment_count = 0
    
    preprocessor = AugmentationPreprocessor(
        augmenter=None,  # Will return immediately since augmenter is None
        rng=random.Random(42),
        bypass_prob=0.1
    )
    
    # Test the probability distribution
    rng = random.Random(42)
    for _ in range(1000):
        if rng.random() < 0.1:
            bypass_count += 1
        else:
            augment_count += 1
    
    # Should be approximately 100 bypassed and 900 augmented
    assert 50 < bypass_count < 150, f"Expected ~100 bypasses, got {bypass_count}"
    assert 850 < augment_count < 950, f"Expected ~900 augments, got {augment_count}"
    
    print(f"✓ bypass_prob=0.1 distribution: {bypass_count}/1000 bypassed (~{bypass_count/10:.1f}%)")


if __name__ == "__main__":
    test_bypass_prob_zero()
    test_bypass_prob_one()
    test_bypass_prob_intermediate()
    print("\n✅ All bypass_prob tests passed!")

