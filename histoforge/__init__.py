"""HistoForge train-your-own deep-learning histopathology platform.

Kept import-light on purpose: importing `histoforge` must not pull the gateway
or vendored ML code. The private deployment helper therefore needs no local
scientific image or model stack.
"""

__version__ = "0.1.4"
