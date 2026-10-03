# The original project's dependencies

The original code was run with these versions, pinned in its
`requirements.txt`. They are kept here as a record, not as an install file:
opencv-python 4.5.3 bundles a libwebp with a known vulnerability
(CVE-2023-4863), and numpy 1.21.0 predates a comparison fix. The rebuilt
package's own dependencies are in `pyproject.toml`.

| Package | Version |
|---|---|
| numpy | 1.21.0 |
| opencv-python | 4.5.3 |
| scikit-image | 0.18.3 |
| matplotlib | 3.4.3 |
