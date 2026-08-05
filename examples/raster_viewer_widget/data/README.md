# Optional local TIFF data

Place both files in this directory to enable the `rr30a` demo dataset:

```text
rr30a_s0_ch1.tif
rr30a_s0_ch2.tif
```

Each file must contain a uint16 TIFF stack shaped `(70, 1024, 1024)` in
`(z, y, x)` order. The demo exposes both an axis-0 maximum projection and the
complete Z stack. The X/Y calibration uses `step=0.15`, `unit="um"`.

TIFF files in this directory are intentionally ignored by Git.
