"""Data pipeline fetch library for processing Argoverse2 LiDAR data."""

from .lidar_processing import (
    create_macro_sweep,
    MacroSweepResult,
    CityName,
    UTM_ZONE_MAP,
    CITY_ORIGIN_LATLONG_DICT,
    MIN_CROP_METERS,
    load_city_annotation,
    parse_city_enum,
)

from .imagery_processing import (
    run_stage_two,
    StageTwoResult,
    DEFAULT_BUFFER_METERS,
)

from .dsm_extraction import (
    extract_dsm_near_lidar,
    DSMExtractionResult,
)

from .gicp_alignment import (
    align_lidar_to_dsm,
    GICPResult,
    GICPParams,
    load_anchor_from_metadata,
)

__all__ = [
    # LiDAR processing
    "create_macro_sweep",
    "MacroSweepResult",
    "CityName",
    "UTM_ZONE_MAP",
    "CITY_ORIGIN_LATLONG_DICT",
    "MIN_CROP_METERS",
    "load_city_annotation",
    "parse_city_enum",
    # Imagery processing
    "run_stage_two",
    "StageTwoResult",
    "DEFAULT_BUFFER_METERS",
    # DSM extraction
    "extract_dsm_near_lidar",
    "DSMExtractionResult",
    # GICP alignment
    "align_lidar_to_dsm",
    "GICPResult",
    "GICPParams",
    "load_anchor_from_metadata",
]
