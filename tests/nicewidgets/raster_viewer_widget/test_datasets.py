"""Tests for synthetic dataset invariants and binary descriptors."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from examples.raster_viewer_widget.datasets import RasterChannel, RasterDataset, RasterHeader, SyntheticDatasetFactory
from examples.raster_viewer_widget.main import DemoRunMode, select_demo_port
from examples.raster_viewer_widget.raster_demo import DemoDatasetCollection
from nicewidgets.raster_viewer_widget.source_registry import REGISTRY, source_plane


def test_every_synthetic_dataset_has_same_shaped_channels() -> None:
    """Verify channel shapes agree within every generated dataset."""
    for dataset in SyntheticDatasetFactory().create_all():
        assert dataset.channels
        assert all(channel.values.shape == dataset.shape for channel in dataset.channels)


def test_descriptor_matches_binary_payloads() -> None:
    """Verify descriptor sizes, dtypes, encodings, and URLs match channel data."""
    collection = DemoDatasetCollection(SyntheticDatasetFactory().create_all())
    for summary in collection.summaries():
        descriptor = collection.descriptor(summary["id"])
        assert descriptor["schema_version"] == "2.0"
        assert descriptor["layout"] == "row-major"
        assert descriptor["endianness"] == "little"
        dataset = collection.get_dataset(summary["id"])
        assert descriptor["header"] == dataset.header.to_json()
        assert descriptor["header"]["shape"] == list(dataset.shape)
        assert descriptor["display_orientation"] == {"transpose": True, "flip_y": True}
        assert "displayOrientation" not in descriptor
        assert len(descriptor["rois"]) == 2
        assert [roi["roi_type"] for roi in descriptor["rois"]] == [
            "rectroi",
            "linesegmentroi",
        ]
        for metadata in descriptor["channels"]:
            channel = collection.get_channel(summary["id"], metadata["id"])
            height, width = dataset.plane_shape
            assert metadata["byte_length"] == height * width * channel.values.dtype.itemsize
            assert "byteLength" not in metadata
            assert "dataUrl" not in metadata
            assert metadata["dtype"] in {"uint16", "float32"}
            assert metadata["encoding"] in {"raw-u16-le", "raw-f32-le"}


def test_binary_plane_response_bypasses_gzip_buffering() -> None:
    """Verify raw pixel responses opt out of NiceGUI's gzip middleware."""
    collection = DemoDatasetCollection(SyntheticDatasetFactory().create_all())
    source = collection.source("square_uint16")
    token = REGISTRY.register(source)
    channel_id = collection.get_dataset("square_uint16").channels[0].channel_id
    try:
        response = source_plane(token, channel_id)
        assert response.headers["content-encoding"] == "identity"
        assert response.headers["content-length"] == str(len(response.body))
    finally:
        REGISTRY.unregister(token)


def test_dataset_summary_reports_channels_shape_and_dtype() -> None:
    """Verify compact summaries contain the requested dataset information."""
    collection = DemoDatasetCollection(SyntheticDatasetFactory().create_all())
    assert collection.summary_text("square_uint16") == "2 channels | 256 × 256 | uint16"
    assert collection.summary_text("float32_features") == "1 channel | 180 × 300 | float32"


def test_display_x_range_uses_transposed_dim_zero_axis() -> None:
    """Verify NiceGUI physical X controls use source dim 0 after transpose."""
    collection = DemoDatasetCollection(SyntheticDatasetFactory().create_all())
    assert collection.display_x_range("linescan_uint16") == pytest.approx((0.0, 30.0))


def test_linescan_shape_and_axis_calibration() -> None:
    """Verify the linescan uses the requested time-by-distance dimensions."""
    dataset = next(
        item
        for item in SyntheticDatasetFactory().create_all()
        if item.dataset_id == "linescan_uint16"
    )
    assert dataset.shape == (30_000, 200)
    assert dataset.header.dims == ("Y", "X")
    assert dataset.header.physical_units == pytest.approx((0.001, 0.2))
    assert dataset.header.physical_units_labels == ("seconds", "um")
    assert dataset.header.sizes == {"Y": 30_000, "X": 200}


def test_single_channel_z_stack_shape_planes_and_calibration() -> None:
    """Verify the generated one-channel Z case exercises calibrated plane loading."""
    dataset = next(
        item
        for item in SyntheticDatasetFactory().create_all()
        if item.dataset_id == "single_channel_z_uint16"
    )
    assert dataset.shape == (20, 512, 512)
    assert dataset.header.dims == ("Z", "Y", "X")
    assert dataset.header.sizes == {"Z": 20, "Y": 512, "X": 512}
    assert dataset.header.num_channels == 1
    assert dataset.header.physical_units == pytest.approx((0.75, 0.4, 0.2))
    assert dataset.header.physical_units_labels == ("um", "um", "um")
    first = dataset.get_plane("channel_0", z_index=0)
    last = dataset.get_plane("channel_0", z_index=19)
    assert first.shape == (512, 512)
    assert first.dtype == np.dtype(np.uint16)
    assert not np.array_equal(first, last)


def test_select_demo_port_uses_preferred_port_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the familiar port is retained when nothing is listening.

    Args:
        monkeypatch: Pytest helper used to isolate the network utility.
    """

    def closed_port(_host: str, _port: int) -> bool:
        """Report that the requested port has no listener.

        Args:
            _host: Ignored host name.
            _port: Ignored TCP port.

        Returns:
            Always ``False``.
        """
        return False

    monkeypatch.setattr("examples.raster_viewer_widget.main.is_port_open", closed_port)
    assert select_demo_port(8080) == 8080


def test_demo_run_mode_defaults_to_web(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify direct module execution retains web-mode compatibility.

    Args:
        monkeypatch: Pytest helper used to isolate the environment.
    """
    monkeypatch.delenv("RASTER_VIEWER_RUN_MODE", raising=False)
    assert DemoRunMode.from_environment() is DemoRunMode.WEB


def test_demo_run_mode_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify invalid launcher modes fail with an actionable error.

    Args:
        monkeypatch: Pytest helper used to isolate the environment.
    """
    monkeypatch.setenv("RASTER_VIEWER_RUN_MODE", "desktop")
    with pytest.raises(ValueError, match="must be 'web' or 'app'"):
        DemoRunMode.from_environment()


def test_select_demo_port_uses_nicegui_free_port_when_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify NiceGUI selects a replacement when the preferred port is busy.

    Args:
        monkeypatch: Pytest helper used to isolate the network utilities.
    """

    def open_port(_host: str, _port: int) -> bool:
        """Report that the requested port already has a listener.

        Args:
            _host: Ignored host name.
            _port: Ignored TCP port.

        Returns:
            Always ``True``.
        """
        return True

    def free_port() -> int:
        """Return a deterministic replacement port.

        Returns:
            Test-only available port.
        """
        return 43123

    monkeypatch.setattr("examples.raster_viewer_widget.main.is_port_open", open_port)
    monkeypatch.setattr("examples.raster_viewer_widget.main.find_free_port", free_port)
    assert select_demo_port(8080) == 43123


def test_channel_normalizes_non_contiguous_uint16() -> None:
    """Verify channels normalize non-contiguous input without changing values."""
    source = np.arange(48, dtype=np.uint16).reshape(6, 8)[:, ::2]
    channel = RasterChannel("test", "Test", source, "gray")
    assert channel.values.flags.c_contiguous
    np.testing.assert_array_equal(channel.values, source)


def test_dataset_rejects_mismatched_channel_shapes() -> None:
    """Verify a dataset cannot contain differently shaped channels."""
    first = RasterChannel("first", "First", np.zeros((4, 5), dtype=np.uint16), "green")
    second = RasterChannel("second", "Second", np.zeros((5, 4), dtype=np.uint16), "magenta")
    header = RasterHeader(
        shape=(4, 5),
        dims=("Y", "X"),
        dtype="uint16",
        num_channels=2,
        physical_units=(1.0, 1.0),
        physical_units_labels=("px", "px"),
    )
    with pytest.raises(ValueError, match="same shape"):
        RasterDataset("bad", "Bad", (first, second), header)


def test_dataset_rejects_mixed_channel_dtypes() -> None:
    """Verify one dataset cannot advertise a misleading shared dtype."""
    first = RasterChannel("first", "First", np.zeros((4, 5), dtype=np.uint16), "green")
    second = RasterChannel("second", "Second", np.zeros((4, 5), dtype=np.float32), "magenta")
    header = RasterHeader(
        shape=(4, 5),
        dims=("Y", "X"),
        dtype="uint16",
        num_channels=2,
        physical_units=(1.0, 1.0),
        physical_units_labels=("px", "px"),
    )
    with pytest.raises(ValueError, match="same dtype"):
        RasterDataset("bad", "Bad", (first, second), header)


def test_z_plane_validation_rejects_invalid_requests() -> None:
    """Verify dimensional and bounds errors are explicit."""
    values = np.zeros((3, 4, 5), dtype=np.uint16)
    channel = RasterChannel("first", "First", values, "green")
    header = RasterHeader(
        shape=(3, 4, 5),
        dims=("Z", "Y", "X"),
        dtype="uint16",
        num_channels=1,
        physical_units=(1.0, 1.0, 1.0),
        physical_units_labels=("slice", "px", "px"),
    )
    dataset = RasterDataset("z", "Z", (channel,), header)
    with pytest.raises(ValueError, match="requires z_index"):
        dataset.get_plane("first")
    with pytest.raises(IndexError, match="outside"):
        dataset.get_plane("first", z_index=3)
    with pytest.raises(ValueError, match="non-negative"):
        dataset.get_plane("first", z_index=0, plus_minus_z=-1)


def test_rr30a_is_absent_when_tiff_files_are_absent(tmp_path: Path) -> None:
    """Verify optional local data does not affect generated demo datasets.

    Args:
        tmp_path: Empty pytest-managed temporary directory.
    """
    datasets = SyntheticDatasetFactory(data_directory=tmp_path).create_all()
    assert "rr30a" not in {dataset.dataset_id for dataset in datasets}


def test_rr30a_loads_two_uint16_max_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify rr30a files become two matching max-projected channels.

    Args:
        tmp_path: Pytest-managed directory used for the expected TIFF names.
        monkeypatch: Pytest helper used to avoid allocating complete TIFF stacks.
    """
    for filename in ("rr30a_s0_ch1.tif", "rr30a_s0_ch2.tif"):
        (tmp_path / filename).touch()
    plane = np.arange(20, dtype=np.uint16).reshape(4, 5)
    stack = np.stack((plane, plane + 100, plane + 200))

    def fake_load_stack(
        _factory: SyntheticDatasetFactory,
        _path: Path,
    ) -> npt.NDArray[np.uint16]:
        """Return a compact synthetic TIFF stack.

        Args:
            _factory: Factory instance receiving the test replacement method.
            _path: Ignored TIFF path passed by the loader.

        Returns:
            Compact three-plane uint16 stack.
        """
        return stack

    monkeypatch.setattr(SyntheticDatasetFactory, "_load_tiff_stack", fake_load_stack)

    datasets = SyntheticDatasetFactory(data_directory=tmp_path).create_all()
    rr30a = next(dataset for dataset in datasets if dataset.dataset_id == "rr30a")
    assert rr30a.shape == (4, 5)
    assert len(rr30a.channels) == 2
    assert rr30a.header.physical_units == pytest.approx((0.15, 0.15))
    assert rr30a.header.physical_units_labels == ("um", "um")
    assert all(channel.values.dtype == np.dtype(np.uint16) for channel in rr30a.channels)
    np.testing.assert_array_equal(rr30a.channels[0].values, plane + 200)

    zstack = next(dataset for dataset in datasets if dataset.dataset_id == "rr30a_zstack")
    assert zstack.shape == (3, 4, 5)
    assert zstack.plane_shape == (4, 5)
    assert zstack.header.dims == ("Z", "Y", "X")
    np.testing.assert_array_equal(zstack.get_plane("channel_1", z_index=1), plane + 100)
    np.testing.assert_array_equal(
        zstack.get_plane("channel_1", z_index=1, plus_minus_z=1),
        plane + 200,
    )
    np.testing.assert_array_equal(
        zstack.get_plane("channel_1", z_index=0, plus_minus_z=1),
        plane + 100,
    )
