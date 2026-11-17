"""Pre-paper names (Engine, Platform) must still resolve, with a DeprecationWarning."""
import pytest

import mist
from mist import ClientType, ClusterConfig, LLMClient, MISTClient, MISTCoordinator


def test_old_names_resolve_to_new_objects():
    with pytest.warns(DeprecationWarning):
        from mist import LLMEngine
    assert LLMEngine is LLMClient

    with pytest.warns(DeprecationWarning):
        from mist.Engine import EngineType
    assert EngineType is ClientType

    with pytest.warns(DeprecationWarning):
        from mist.Platforms.platforms import PlatformConfig
    assert PlatformConfig is ClusterConfig

    with pytest.raises(AttributeError):
        mist.NoSuchEngine


def test_old_kwargs_attributes_and_methods():
    with pytest.warns(DeprecationWarning):
        host = MISTClient(model=None, engine_types=[ClientType.HOST], engine_id=7)
    assert host.client_types == [ClientType.HOST] and host.client_id == 7

    with pytest.warns(DeprecationWarning):
        assert host.engine_types == [ClientType.HOST]

    coordinator = MISTCoordinator([], logging_file=None)
    with pytest.warns(DeprecationWarning):
        assert coordinator.add_engine == coordinator.add_client


def test_assigning_old_attribute_names_sets_new_ones():
    from mist import Request
    from mist.Request.request import StageMetrics

    req = Request(input_len=8, output_len=2)
    with pytest.warns(DeprecationWarning):
        req.engine_preference = [3]
    assert req.client_preference == [3] and "engine_preference" not in vars(req)

    stage = StageMetrics()
    with pytest.warns(DeprecationWarning):
        stage.engine_exit_time = 5.0
    assert stage.client_exit_time == 5.0
