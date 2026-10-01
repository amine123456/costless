"""CI integration: fetch the baseline run and post the report on the MR / PR."""

from costless.ci.platforms import CIError, CIPlatform, detect_platform

__all__ = ["CIError", "CIPlatform", "detect_platform"]
