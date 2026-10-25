# HistoForge backend

The cloud backend and local setup helper for HistoForge, a tissue-analysis
platform for QuPath. It runs model inference and fine-tuning in your own Modal
workspace. Analysis modules provide models, classes, and measurement scripts
through Hugging Face.

## Setup

See the [HistoForge documentation](https://histoforge.github.io/docs/) for installation
and usage.

## Supported model architectures

| Architecture | Prediction head | Output |
| --- | --- | --- |
| DINOv3 ViT-L/16 + spatial adapter | Cellpose flows | Instance segmentation, with optional object classification |
| DINOv3 ViT-L/16 + spatial adapter | HoVer-Net horizontal/vertical distances | Instance segmentation, with optional object classification |
| DINOv3 ViT-L/16 + spatial adapter | Pixel classifier | Semantic segmentation (pixel classification) |

## Development

- [Environment setup and tests](TESTING.md)
- [Module format and contracts](custom_docs/contracts/modules.md)
- [Developer tools and dependency maintenance](scripts/README.md)

## License

See [LICENSE](LICENSE). Vendored code, model weights, and datasets retain their
own licenses and attribution.
