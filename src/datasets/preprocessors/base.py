"""Base preprocessor following ms-swift patterns"""
from typing import Any, Dict, Optional


class BasePreprocessor:
    """Base preprocessor for row-level transformations.
    
    Follows ms-swift RowPreprocessor pattern for pluggable, composable preprocessing.
    """
    
    def __init__(self, **kwargs):
        """Initialize preprocessor with optional configuration."""
        self.config = kwargs
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single row.
        
        Args:
            row: Input row dictionary
            
        Returns:
            Processed row dictionary, or None to skip this row
        """
        return row
    
    def __call__(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Allow preprocessor to be called as a function."""
        return self.preprocess(row)


__all__ = ["BasePreprocessor"]

