"""Format and clean annotation descriptions for data_conversion."""

from typing import Any, Dict

from data_conversion.annotation_cleaner import clean_annotation_content
from data_conversion.utils.sanitizers import (
    sanitize_desc_value,
    sanitize_free_text_value,
)


class FormatConverter:
    """Handles conversion between different data formats."""

    @staticmethod
    def format_description(
        content_dict: Dict[str, str],
    ) -> str:
        """Format content dictionary to key=value description string."""
        obj_type = str(content_dict.get("object_type", "")).strip()
        prop = str(content_dict.get("property", "")).strip()
        extra = str(content_dict.get("extra_info", "")).strip()

        if not obj_type:
            return ""

        pairs = [("类别", sanitize_desc_value(obj_type) or obj_type)]
        if prop:
            pairs.append(("属性", sanitize_desc_value(prop) or prop))
        if extra:
            pairs.append(("备注", sanitize_free_text_value(extra) or extra))

        return ",".join(f"{k}={v}" for k, v in pairs if v)

    @staticmethod
    def parse_description_string(description: str) -> Dict[str, str]:
        """Parse description string back into components (key=value or legacy)."""
        components = {"object_type": "", "property": "", "extra_info": ""}

        if not description:
            return components

        tokens = [t.strip() for t in description.split(",") if t.strip()]
        if any("=" in t for t in tokens):
            current_key = None
            for token in tokens:
                if "=" in token:
                    key, value = token.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    current_key = key
                    if key == "类别":
                        components["object_type"] = value
                    elif key == "属性":
                        components["property"] = value
                    elif key == "备注":
                        components["extra_info"] = value
                else:
                    if current_key == "备注" or not current_key:
                        if components["extra_info"]:
                            components["extra_info"] = (
                                f"{components['extra_info']},{token}"
                            )
                        else:
                            components["extra_info"] = token
        else:
            # Legacy slash-delimited format
            parts = description.split("/")
            if len(parts) >= 1:
                components["object_type"] = parts[0].strip()
            if len(parts) >= 2:
                components["property"] = parts[1].strip()
            if len(parts) >= 3:
                components["extra_info"] = "/".join(parts[2:]).strip()

        return components

    @staticmethod
    def clean_annotation_content(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean annotation content preserving essential Chinese structure only."""
        return clean_annotation_content(data, lang="zh")
