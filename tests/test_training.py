from self_healing_embodied_agents.training import train_transition_model


def test_small_training_run_is_finite() -> None:
    bundle, metrics = train_transition_model(seed=1, episodes=30, epochs=10)
    assert metrics.validation_mse >= 0.0
    assert metrics.residual_threshold > 0.0
    assert bundle.model is not None
