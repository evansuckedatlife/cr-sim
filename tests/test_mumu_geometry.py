"""Geometry is the part of cr_sim.mumu that needs no device to test, so it
gets tested thoroughly here: the homography must reproduce its own four
calibration corners exactly, round-trip screen -> tile -> screen within a
pixel, and reject tiles that fall outside the 18x32 board.
"""

from __future__ import annotations

import pytest

from cr_sim.mumu.geometry import (
    ARENA_HEIGHT_TILES,
    ARENA_WIDTH_TILES,
    DEFAULT_CALIBRATION,
    ArenaCalibration,
    CoordinateMapper,
    TileOutOfBoundsError,
    calibrate_from_corners,
    require_tile_in_bounds,
    tile_in_bounds,
)


def test_arena_dimensions_match_the_engine():
    # 18 wide x 32 tall is confirmed in reference/anchors.json
    # (engine_constants.arena_tiles_wide / arena_tiles_tall) and read here
    # from cr_sim.engine.arena rather than duplicated.
    assert ARENA_WIDTH_TILES == 18.0
    assert ARENA_HEIGHT_TILES == 32.0


class TestHomographyExactness:
    """The DLT solve must reproduce its own 4 correspondences exactly (up to
    floating point), for both the tile->pixel and pixel->tile directions."""

    def test_default_calibration_corners_map_to_the_given_pixels(self):
        mapper = CoordinateMapper(DEFAULT_CALIBRATION)
        cases = [
            ((0.0, 0.0), DEFAULT_CALIBRATION.top_left),
            ((ARENA_WIDTH_TILES, 0.0), DEFAULT_CALIBRATION.top_right),
            ((0.0, ARENA_HEIGHT_TILES), DEFAULT_CALIBRATION.bottom_left),
            ((ARENA_WIDTH_TILES, ARENA_HEIGHT_TILES), DEFAULT_CALIBRATION.bottom_right),
        ]
        for (tile_x, tile_y), expected_px in cases:
            px, py = mapper.tile_to_screen(tile_x, tile_y)
            assert px == pytest.approx(expected_px[0], abs=1e-6)
            assert py == pytest.approx(expected_px[1], abs=1e-6)

    def test_pixel_corners_map_back_to_the_tile_corners(self):
        mapper = CoordinateMapper(DEFAULT_CALIBRATION)
        cases = [
            (DEFAULT_CALIBRATION.top_left, (0.0, 0.0)),
            (DEFAULT_CALIBRATION.top_right, (ARENA_WIDTH_TILES, 0.0)),
            (DEFAULT_CALIBRATION.bottom_left, (0.0, ARENA_HEIGHT_TILES)),
            (DEFAULT_CALIBRATION.bottom_right, (ARENA_WIDTH_TILES, ARENA_HEIGHT_TILES)),
        ]
        for (px, py), expected_tile in cases:
            tile_x, tile_y = mapper.screen_to_tile(px, py)
            assert tile_x == pytest.approx(expected_tile[0], abs=1e-6)
            assert tile_y == pytest.approx(expected_tile[1], abs=1e-6)

    def test_arbitrary_quadrilateral_also_solves_exactly(self):
        # Not the default trapezoid -- an irregular quad, to check the DLT
        # solve isn't secretly relying on any symmetry in DEFAULT_CALIBRATION.
        mapper = calibrate_from_corners(
            top_left=(50.0, 30.0),
            top_right=(900.0, 60.0),
            bottom_left=(-20.0, 1400.0),
            bottom_right=(1000.0, 1350.0),
            screen_width=1080,
            screen_height=1920,
        )
        px, py = mapper.tile_to_screen(ARENA_WIDTH_TILES, 0.0)
        assert (px, py) == pytest.approx((900.0, 60.0), abs=1e-6)


class TestRoundTrip:
    """screen -> tile -> screen must land within a pixel; corners map to
    corners exactly, and interior points close the loop cleanly."""

    @pytest.mark.parametrize(
        "px,py",
        [
            (500.0, 800.0),
            (300.0, 400.0),
            (900.0, 1200.0),
            (150.0, 1500.0),
            (1000.0, 200.0),
        ],
    )
    def test_screen_tile_screen_round_trip_within_a_pixel(self, px, py):
        mapper = CoordinateMapper(DEFAULT_CALIBRATION)
        tile_x, tile_y = mapper.screen_to_tile(px, py)
        px2, py2 = mapper.tile_to_screen(tile_x, tile_y)
        assert px2 == pytest.approx(px, abs=1.0)
        assert py2 == pytest.approx(py, abs=1.0)

    @pytest.mark.parametrize(
        "tile_x,tile_y",
        [(0.0, 0.0), (9.0, 16.0), (18.0, 32.0), (3.5, 6.5), (14.5, 25.5)],
    )
    def test_tile_screen_tile_round_trip_within_a_hundredth_tile(self, tile_x, tile_y):
        mapper = CoordinateMapper(DEFAULT_CALIBRATION)
        px, py = mapper.tile_to_screen(tile_x, tile_y)
        tile_x2, tile_y2 = mapper.screen_to_tile(px, py)
        assert tile_x2 == pytest.approx(tile_x, abs=0.01)
        assert tile_y2 == pytest.approx(tile_y, abs=0.01)


class TestSubtileConversion:
    """tile <-> subtile must agree with cr_sim.engine.fixed's own rounding,
    since that is the single source of truth this module delegates to."""

    def test_tile_to_subtile_matches_engine_fixed(self):
        from cr_sim.engine.fixed import SUBTILES_PER_TILE

        mapper = CoordinateMapper(DEFAULT_CALIBRATION)
        x, y = mapper.tile_to_subtile(1.0, 1.0)
        assert (x, y) == (SUBTILES_PER_TILE, SUBTILES_PER_TILE)

    def test_subtile_to_tile_is_the_inverse(self):
        from cr_sim.engine.fixed import SUBTILES_PER_TILE

        mapper = CoordinateMapper(DEFAULT_CALIBRATION)
        tile_x, tile_y = mapper.subtile_to_tile(SUBTILES_PER_TILE * 9, SUBTILES_PER_TILE * 16)
        assert tile_x == pytest.approx(9.0)
        assert tile_y == pytest.approx(16.0)

    def test_screen_to_subtile_and_back_round_trips(self):
        mapper = CoordinateMapper(DEFAULT_CALIBRATION)
        x, y = mapper.screen_to_subtile(500.0, 800.0)
        px, py = mapper.subtile_to_screen(x, y)
        assert px == pytest.approx(500.0, abs=1.0)
        assert py == pytest.approx(800.0, abs=1.0)


class TestBounds:
    def test_arena_corners_are_in_bounds(self):
        assert tile_in_bounds(0.0, 0.0)
        assert tile_in_bounds(ARENA_WIDTH_TILES, ARENA_HEIGHT_TILES)

    @pytest.mark.parametrize(
        "tile_x,tile_y",
        [(-0.01, 5.0), (5.0, -0.01), (ARENA_WIDTH_TILES + 0.01, 5.0), (5.0, ARENA_HEIGHT_TILES + 0.01)],
    )
    def test_out_of_bounds_tiles_are_rejected(self, tile_x, tile_y):
        assert not tile_in_bounds(tile_x, tile_y)
        with pytest.raises(TileOutOfBoundsError):
            require_tile_in_bounds(tile_x, tile_y)

    def test_in_bounds_tile_does_not_raise(self):
        require_tile_in_bounds(9.0, 16.0)  # must not raise


class TestCalibrationSerialization:
    def test_json_round_trip(self, tmp_path):
        path = tmp_path / "calibration.json"
        DEFAULT_CALIBRATION.save(path)
        loaded = ArenaCalibration.load(path)
        assert loaded == DEFAULT_CALIBRATION

    def test_from_dict_defaults_an_absent_label(self):
        data = DEFAULT_CALIBRATION.to_dict()
        del data["label"]
        loaded = ArenaCalibration.from_dict(data)
        assert loaded.label == "unlabelled"

    def test_calibrate_from_corners_matches_manual_construction(self):
        via_helper = calibrate_from_corners(
            top_left=(10.0, 10.0),
            top_right=(500.0, 20.0),
            bottom_left=(0.0, 800.0),
            bottom_right=(510.0, 790.0),
            screen_width=520,
            screen_height=820,
            label="test",
        )
        manual = CoordinateMapper(
            ArenaCalibration(
                top_left=(10.0, 10.0),
                top_right=(500.0, 20.0),
                bottom_left=(0.0, 800.0),
                bottom_right=(510.0, 790.0),
                screen_width=520,
                screen_height=820,
                label="test",
            )
        )
        assert via_helper.tile_to_screen(9.0, 16.0) == pytest.approx(
            manual.tile_to_screen(9.0, 16.0), abs=1e-9
        )
