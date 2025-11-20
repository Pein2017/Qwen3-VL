"""Format and clean annotation descriptions for data_conversion."""

from typing import Dict

from data_conversion.annotation_cleaner import clean_annotation_content


class FormatConverter:
    """Handles conversion between different data formats."""

    @staticmethod
    def format_description(
        content_dict: Dict[str, str],
    ) -> str:
        """Format content dictionary to Chinese description string."""
        parts = []
        for resp_type in ("object_type", "property", "extra_info"):
            value = content_dict.get(resp_type, "")
            if value:
                parts.append(str(value))

        if not parts:
            return ""

        # Use compact format for Chinese
        result = "/".join(parts)
        return result.replace(", ", "/").replace(",", "/")

    @staticmethod
    def parse_description_string(description: str) -> Dict[str, str]:
        """Parse Chinese description string back into components."""
        components = {"object_type": "", "property": "", "extra_info": ""}

        if not description:
            return components

        # Parse compact format (Chinese)
        parts = description.split("/")
        if len(parts) >= 1:
            components["object_type"] = parts[0].strip()
        if len(parts) >= 2:
            components["property"] = parts[1].strip()
        if len(parts) >= 3:
            components["extra_info"] = "/".join(parts[2:]).strip()

        return components

    @staticmethod
    def clean_annotation_content(data: Dict) -> Dict:
        """Clean annotation content preserving essential Chinese structure only."""
        return clean_annotation_content(data, lang="zh")
