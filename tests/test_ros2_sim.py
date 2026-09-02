"""Tests for src/connectors/ros2_sim.py — pure unit tests with a temp state file."""

import json
import os
import tempfile

import pytest

import src.connectors.ros2_sim as sim


@pytest.fixture(autouse=True)
def _tmp_state(tmp_path, monkeypatch):
    """Redirect the simulator's state file to a temporary path per test."""
    state_file = str(tmp_path / "robot_state.json")
    monkeypatch.setattr(sim, "_STATE_FILE", state_file)
    yield state_file


class TestDefaultState:
    def test_default_room_is_hallway(self):
        state = sim.get_state()
        assert state["room"] == "hallway"

    def test_default_battery(self):
        state = sim.get_state()
        assert state["battery"] == 92

    def test_default_status_is_idle(self):
        state = sim.get_state()
        assert state["status"] == "idle"

    def test_status_string_contains_hallway(self):
        text = sim.sim_status()
        assert "Hallway" in text

    def test_status_string_contains_battery(self):
        text = sim.sim_status()
        assert "92%" in text

    def test_status_string_mentions_simulator(self):
        text = sim.sim_status()
        assert "Simulator" in text or "simulator" in text


class TestBFSNavigation:
    def test_navigate_same_room_no_move(self):
        result = sim.sim_navigate("hallway")
        assert "Already in" in result

    def test_navigate_to_bedroom(self):
        result = sim.sim_navigate("bedroom")
        assert "Bedroom" in result
        assert "Route" in result
        state = sim.get_state()
        assert state["room"] == "bedroom"

    def test_navigate_to_kitchen(self):
        result = sim.sim_navigate("kitchen")
        # Hallway → Office → Kitchen
        assert "Kitchen" in result
        state = sim.get_state()
        assert state["room"] == "kitchen"

    def test_navigate_multi_hop_path(self):
        # kitchen is 2 hops from hallway: hallway → office → kitchen
        result = sim.sim_navigate("kitchen")
        assert "Hallway" in result or "Office" in result

    def test_navigate_drains_battery(self):
        initial = sim.get_state()["battery"]
        sim.sim_navigate("bedroom")  # 1 hop
        after = sim.get_state()["battery"]
        assert after < initial

    def test_battery_floor_is_5(self):
        # Force low battery
        state = sim.get_state()
        state["battery"] = 5
        sim._save(state)
        sim.sim_navigate("bedroom")
        after = sim.get_state()["battery"]
        assert after >= 5

    def test_navigate_unknown_room_returns_error(self):
        result = sim.sim_navigate("moon_base")
        assert "ERROR" in result or "unknown" in result

    def test_navigate_updates_last_action(self):
        sim.sim_navigate("office")
        state = sim.get_state()
        assert state["last_action"] is not None
        assert "Navigated" in state["last_action"]

    def test_state_persists_after_navigate(self):
        sim.sim_navigate("bedroom")
        # Re-load from the file (simulate a fresh process)
        reloaded = sim._load()
        assert reloaded["room"] == "bedroom"


class TestDescribeScene:
    def test_describe_hallway(self):
        result = sim.sim_describe_scene()
        assert "Hallway" in result
        assert "front door" in result.lower() or "hallway" in result.lower()

    def test_describe_after_navigate(self):
        sim.sim_navigate("kitchen")
        result = sim.sim_describe_scene()
        assert "Kitchen" in result
        assert "refrigerator" in result.lower() or "stove" in result.lower()

    def test_describe_includes_battery(self):
        result = sim.sim_describe_scene()
        assert "%" in result

    def test_all_rooms_have_descriptions(self):
        for room in sim.ROOMS:
            state = sim._load()
            state["room"] = room
            sim._save(state)
            result = sim.sim_describe_scene()
            assert room.replace("_", " ").title() in result


class TestFineMove:
    def test_move_stays_in_room(self):
        result = sim.sim_move("forward", 0.5)
        assert "Hallway" in result
        assert sim.get_state()["room"] == "hallway"

    def test_move_drains_battery(self):
        initial = sim.get_state()["battery"]
        sim.sim_move("forward", 3.0)
        after = sim.get_state()["battery"]
        assert after < initial


class TestCancel:
    def test_cancel_sets_idle(self):
        result = sim.sim_cancel()
        assert "IDLE" in result or "stop" in result.lower()
        assert sim.get_state()["status"] == "idle"

    def test_cancel_records_action(self):
        sim.sim_cancel()
        assert "stop" in sim.get_state()["last_action"].lower()
