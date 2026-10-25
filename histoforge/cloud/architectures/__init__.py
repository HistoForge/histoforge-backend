"""Per-architecture Modal runners.

One runner class per arm-b decoder head (cellpose flow / hovernet HV / semantic
DS). Each runner is parameterized by `ref` (`modal.parameter()`, qualified
`<owner>/<name>`), reads the matching manifest from the Volume, builds the
vendored net, and serves inference.

Registry mapping (architecture string → runner Cls name) lives in `registry.py`.
"""
