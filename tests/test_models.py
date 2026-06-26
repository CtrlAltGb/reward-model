"""Tests for mock model implementations — assert model loads exactly once."""

from __future__ import annotations

import numpy as np

from rdf.models.mock import (
    MockDeminfModel,
    MockRobometerModel,
    get_init_call_count,
    reset_init_call_count,
)


class TestMockRobometerModel:
    def setup_method(self):
        reset_init_call_count()

    def test_loads_once(self):
        model = MockRobometerModel()
        assert get_init_call_count() == 1

    def test_scores_episode(self, mock_robometer):
        score = mock_robometer.score_episode("/tmp/test.mp4", "pick up the cup")
        assert 0.0 <= score.reward <= 1.0
        assert 0.0 <= score.success_pred <= 1.0
        assert score.frames_used > 0

    def test_deterministic(self, mock_robometer):
        s1 = mock_robometer.score_episode("/tmp/test.mp4", "task A")
        s2 = mock_robometer.score_episode("/tmp/test.mp4", "task A")
        assert s1.reward == s2.reward
        assert s1.success_pred == s2.success_pred

    def test_different_inputs_different_scores(self, mock_robometer):
        s1 = mock_robometer.score_episode("/tmp/test.mp4", "task A")
        s2 = mock_robometer.score_episode("/tmp/test.mp4", "task B")
        assert s1.reward != s2.reward

    def test_scored_twice_model_loaded_once(self):
        reset_init_call_count()
        model = MockRobometerModel()
        model.score_episode("/tmp/test.mp4", "task A")
        model.score_episode("/tmp/test.mp4", "task A")
        assert get_init_call_count() == 1


class TestMockDeminfModel:
    def setup_method(self):
        reset_init_call_count()

    def test_loads_once(self):
        model = MockDeminfModel()
        assert get_init_call_count() == 1

    def test_encode_episodes(self, mock_deminf):
        states = [np.random.random((10, 64)).astype(np.float32) for _ in range(3)]
        actions = [np.random.random((10, 7)).astype(np.float32) for _ in range(3)]
        obs_lat, act_lat = mock_deminf.encode_episodes(states, actions)
        assert obs_lat.shape == (3, MockDeminfModel.LATENT_DIM)
        assert act_lat.shape == (3, MockDeminfModel.LATENT_DIM)

    def test_score_against_reference(self, mock_deminf):
        states = [np.random.random((10, 64)).astype(np.float32) for _ in range(5)]
        actions = [np.random.random((10, 7)).astype(np.float32) for _ in range(5)]
        obs_lat, act_lat = mock_deminf.encode_episodes(states, actions)
        ref_obs = np.zeros((10, MockDeminfModel.LATENT_DIM), dtype=np.float32)
        ref_act = np.zeros((10, MockDeminfModel.LATENT_DIM), dtype=np.float32)
        scores = mock_deminf.score_against_reference(obs_lat, act_lat, ref_obs, ref_act)
        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)

    def test_model_loaded_once_scored_many(self):
        reset_init_call_count()
        model = MockDeminfModel()
        for _ in range(3):
            states = [np.zeros((5, 64), dtype=np.float32)]
            actions = [np.zeros((5, 7), dtype=np.float32)]
            model.encode_episodes(states, actions)
        assert get_init_call_count() == 1
