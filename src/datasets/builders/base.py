"""Base builder interface"""
from typing import Any, Dict


class BaseBuilder:
    """Base class for message builders.
    
    Builders construct messages from paired records for training.
    """
    
    def __init__(self, **kwargs):
        """Initialize builder with configuration."""
        self.config = kwargs
    
    def build(self, record_a: Dict[str, Any], record_b: Dict[str, Any]) -> Dict[str, Any]:
        """Build messages from two records.
        
        Args:
            record_a: First record
            record_b: Second record (may be same as record_a)
            
        Returns:
            Dictionary with 'messages', 'images', and optional 'objects' keys
        """
        raise NotImplementedError("Subclasses must implement build()")
    
    # Optional: for N-way grouping
    def build_many(self, records: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            raise ValueError('build_many requires at least one record')
        if len(records) == 1:
            return self.build(records[0], records[0])
        return self.build(records[0], records[1])
    
    def __call__(self, record_a: Dict[str, Any], record_b: Dict[str, Any]) -> Dict[str, Any]:
        """Allow builder to be called as a function."""
        return self.build(record_a, record_b)


__all__ = ["BaseBuilder"]

