"""Tests for src/tool_registry.py — pure unit tests, no external deps."""

import pytest
from src.tool_registry import (
    READ, WRITE, DESTRUCTIVE,
    TOOL_TIERS, WRITE_TOOLS, DESTRUCTIVE_TOOLS,
    tier_of, is_write, is_destructive,
)


class TestConstants:
    def test_tier_values_are_strings(self):
        assert isinstance(READ, str)
        assert isinstance(WRITE, str)
        assert isinstance(DESTRUCTIVE, str)

    def test_tier_values_are_distinct(self):
        assert len({READ, WRITE, DESTRUCTIVE}) == 3


class TestToolTiers:
    def test_all_values_are_valid_tiers(self):
        valid = {READ, WRITE, DESTRUCTIVE}
        for name, tier in TOOL_TIERS.items():
            assert tier in valid, f"{name!r} has unknown tier {tier!r}"

    def test_read_only_tools(self):
        read_only = ["memory_list", "get_calendar_events", "web_search",
                     "robot_status", "robot_describe_scene"]
        for t in read_only:
            assert tier_of(t) == READ, f"{t} should be READ"

    def test_write_tools(self):
        write_tools = ["memory_remember", "add_task", "create_event",
                       "write_local_file", "notes_create"]
        for t in write_tools:
            assert tier_of(t) == WRITE, f"{t} should be WRITE"

    def test_destructive_tools(self):
        destructive = ["send_email", "imessage_send", "control_device",
                       "robot_move", "robot_navigate", "telegram_send",
                       "slack_send_message", "car_lock"]
        for t in destructive:
            assert tier_of(t) == DESTRUCTIVE, f"{t} should be DESTRUCTIVE"

    def test_unknown_tool_defaults_to_write(self):
        assert tier_of("nonexistent_tool_xyz") == WRITE


class TestDerivedSets:
    def test_write_tools_is_frozenset(self):
        assert isinstance(WRITE_TOOLS, frozenset)

    def test_destructive_tools_is_frozenset(self):
        assert isinstance(DESTRUCTIVE_TOOLS, frozenset)

    def test_destructive_subset_of_write(self):
        assert DESTRUCTIVE_TOOLS.issubset(WRITE_TOOLS)

    def test_read_tools_not_in_write(self):
        read_only = {"memory_list", "get_calendar_events", "web_search"}
        assert read_only.isdisjoint(WRITE_TOOLS)

    def test_write_tools_non_empty(self):
        assert len(WRITE_TOOLS) > 0

    def test_destructive_tools_non_empty(self):
        assert len(DESTRUCTIVE_TOOLS) > 0


class TestHelpers:
    def test_is_write_returns_true_for_write(self):
        assert is_write("memory_remember") is True

    def test_is_write_returns_true_for_destructive(self):
        assert is_write("send_email") is True

    def test_is_write_returns_false_for_read(self):
        assert is_write("web_search") is False

    def test_is_destructive_true(self):
        assert is_destructive("robot_move") is True
        assert is_destructive("car_lock") is True

    def test_is_destructive_false_for_write(self):
        assert is_destructive("add_task") is False

    def test_is_destructive_false_for_read(self):
        assert is_destructive("memory_list") is False

    def test_unknown_tool_is_write(self):
        assert is_write("unknown_xyz") is True
        assert is_destructive("unknown_xyz") is False

    def test_robot_describe_scene_is_read(self):
        assert tier_of("robot_describe_scene") == READ
        assert is_write("robot_describe_scene") is False

    def test_robot_navigate_is_destructive(self):
        assert tier_of("robot_navigate") == DESTRUCTIVE
        assert is_destructive("robot_navigate") is True
