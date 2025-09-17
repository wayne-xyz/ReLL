# ReLL

## AV2 Pose Coordinate Export
- Script: `ArgoverseLidar/export_pose_coordinates.py` enumerates the public Argoverse bucket, streams every `city_SE3_egovehicle.feather` over HTTPS, and writes a consolidated Arrow feather without downloading the full dataset.
- Run `python ArgoverseLidar/export_pose_coordinates.py --output av2_coor.feather` (options: `--splits`, `--limit-logs`, `--remote-prefix`, `--max-keys`). Progress lines show which log is being processed and the cumulative row count.
- Each row in `av2_coor.feather` carries `city`, `split`, `log_id`, `timestamp_ns`, quaternion (`qw`, `qx`, `qy`, `qz`), translation (`tx`, `ty`, `tz`), UTM info (`zone_num_hemi`, `easting`, `northing`), and WGS84 coordinates (`latitude`, `longitude`).
- City codes are inferred from map filenames, mapped to their UTM zones, and pose translations are projected to UTM/WGS84 using `pyproj`.
