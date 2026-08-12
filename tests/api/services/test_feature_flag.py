"""
Unit tests for the USE_RECONCILED_STRUCTURE feature flag behavior.

Tests cover:
- Default value is false when env var is not set
- Truthy values (true, 1, yes) are parsed correctly
- Falsy values (false, 0, no) are parsed correctly
- Unrecognized values default to false
- Dynamic toggling without restart (change env var, verify next read reflects change)

**Validates: Requirements 6.1, 6.4, 6.5**
"""

import os
import pytest

from src.core.config import Settings


# ======================================================================
# Default behavior
# ======================================================================

class TestFeatureFlagDefault:
    """When USE_RECONCILED_STRUCTURE is not set, default to false."""

    def test_default_is_false_when_env_var_not_set(self):
        env = os.environ.pop("USE_RECONCILED_STRUCTURE", None)
        try:
            s = Settings()
            assert s.use_reconciled_structure is False
        finally:
            if env is not None:
                os.environ["USE_RECONCILED_STRUCTURE"] = env


# ======================================================================
# Truthy values
# ======================================================================

class TestFeatureFlagTruthyValues:
    """true, 1, yes (case-insensitive) should all resolve to True."""

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "tRuE"])
    def test_true_variants(self, value):
        os.environ["USE_RECONCILED_STRUCTURE"] = value
        try:
            s = Settings()
            assert s.use_reconciled_structure is True
        finally:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)

    def test_one_is_true(self):
        os.environ["USE_RECONCILED_STRUCTURE"] = "1"
        try:
            s = Settings()
            assert s.use_reconciled_structure is True
        finally:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)

    @pytest.mark.parametrize("value", ["yes", "Yes", "YES"])
    def test_yes_variants(self, value):
        os.environ["USE_RECONCILED_STRUCTURE"] = value
        try:
            s = Settings()
            assert s.use_reconciled_structure is True
        finally:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)


# ======================================================================
# Falsy values
# ======================================================================

class TestFeatureFlagFalsyValues:
    """false, 0, no should all resolve to False."""

    @pytest.mark.parametrize("value", ["false", "False", "FALSE"])
    def test_false_variants(self, value):
        os.environ["USE_RECONCILED_STRUCTURE"] = value
        try:
            s = Settings()
            assert s.use_reconciled_structure is False
        finally:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)

    def test_zero_is_false(self):
        os.environ["USE_RECONCILED_STRUCTURE"] = "0"
        try:
            s = Settings()
            assert s.use_reconciled_structure is False
        finally:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)

    @pytest.mark.parametrize("value", ["no", "No", "NO"])
    def test_no_variants(self, value):
        os.environ["USE_RECONCILED_STRUCTURE"] = value
        try:
            s = Settings()
            assert s.use_reconciled_structure is False
        finally:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)


# ======================================================================
# Unrecognized values
# ======================================================================

class TestFeatureFlagUnrecognizedValues:
    """Unrecognized values should default to false (safe fallback)."""

    @pytest.mark.parametrize("value", ["maybe", "on", "off", "enabled", "2", ""])
    def test_unrecognized_defaults_to_false(self, value):
        os.environ["USE_RECONCILED_STRUCTURE"] = value
        try:
            s = Settings()
            assert s.use_reconciled_structure is False
        finally:
            os.environ.pop("USE_RECONCILED_STRUCTURE", None)


# ======================================================================
# Dynamic toggling without restart
# ======================================================================

class TestFeatureFlagDynamicToggling:
    """Changing the env var should be reflected on the next property read."""

    def test_toggle_false_to_true(self):
        os.environ["USE_RECONCILED_STRUCTURE"] = "false"
        s = Settings()
        assert s.use_reconciled_structure is False

        os.environ["USE_RECONCILED_STRUCTURE"] = "true"
        # Same instance, next read should reflect the change
        assert s.use_reconciled_structure is True

    def test_toggle_true_to_false(self):
        os.environ["USE_RECONCILED_STRUCTURE"] = "true"
        s = Settings()
        assert s.use_reconciled_structure is True

        os.environ["USE_RECONCILED_STRUCTURE"] = "false"
        assert s.use_reconciled_structure is False

    def test_toggle_through_multiple_values(self):
        s = Settings()

        for value, expected in [("false", False), ("true", True), ("1", True),
                                ("0", False), ("yes", True), ("no", False)]:
            os.environ["USE_RECONCILED_STRUCTURE"] = value
            assert s.use_reconciled_structure is expected, (
                f"Expected {expected} for value {value!r}, "
                f"got {s.use_reconciled_structure}"
            )

    def test_unset_after_set_returns_false(self):
        os.environ["USE_RECONCILED_STRUCTURE"] = "true"
        s = Settings()
        assert s.use_reconciled_structure is True

        os.environ.pop("USE_RECONCILED_STRUCTURE", None)
        assert s.use_reconciled_structure is False

    def teardown_method(self):
        """Clean up env var after each test."""
        os.environ.pop("USE_RECONCILED_STRUCTURE", None)
