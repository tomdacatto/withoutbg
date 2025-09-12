# withoutbg

**AI-powered background removal with local and cloud options**

[![PyPI](https://img.shields.io/pypi/v/withoutbg.svg)](https://pypi.org/project/withoutbg/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/withoutbg/withoutbg/actions/workflows/ci.yml/badge.svg)](https://github.com/withoutbg/withoutbg/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/withoutbg/withoutbg/branch/main/graph/badge.svg)](https://codecov.io/gh/withoutbg/withoutbg)

Remove backgrounds from images instantly with AI. Choose between local processing (free) or cloud API (best quality).

## 🚀 Quick Start

```bash
# Install
pip install withoutbg

# Remove background (local processing)
withoutbg image.jpg

# Use cloud API for best quality processing
withoutbg image.jpg --api-key sk_your_api_key
```

## ✨ Visual Examples

See the power of AI background removal in action:

![Woman with confetti - before and after](examples/images/woman-golden-hour.jpg)

![Dog portrait - before and after](examples/images/dog-mugshot.jpg)

![Pizza product - before and after](examples/images/pizza-parody.jpg)

![Feather detail - before and after](examples/images/feather.jpg)

*Perfect edge detection, hair details, and transparent backgrounds*


## 💻 Python API

```python
from withoutbg import remove_background

# Local processing with Snap model (free)
result = remove_background("input.jpg")
result.save("output.png")

# Cloud processing with API (best quality)
result = remove_background("input.jpg", api_key="sk_your_key")

# Batch processing
from withoutbg import remove_background_batch
results = remove_background_batch(["img1.jpg", "img2.jpg"], 
                                  output_dir="results/")
```

## 🖥️ CLI Usage

### Basic Usage
```bash
# Process single image
withoutbg photo.jpg

# Specify output path
withoutbg photo.jpg --output result.png

# Use different format
withoutbg photo.jpg --format webp --quality 90
```

### Cloud API 
```bash
# Set API key via environment
export WITHOUTBG_API_KEY="sk_your_api_key"
withoutbg photo.jpg --use-api

# Or pass directly
withoutbg photo.jpg --api-key sk_your_key
```

### Batch Processing
```bash
# Process all images in directory
withoutbg photos/ --batch --output-dir results/

# With cloud API for best quality
withoutbg photos/ --batch --use-api --output-dir results/
```

## 🔧 Installation Options

### Standard Installation
```bash
pip install withoutbg
```

### Development
```bash
git clone https://github.com/withoutbg/withoutbg.git
cd withoutbg
pip install -e ".[dev]"
```

## 🎨 Examples

### Basic Background Removal
```python
import withoutbg

# Simple usage
output = withoutbg.remove_background("portrait.jpg")
output.save("portrait-withoutbg.png")
```

### E-commerce Product Photos
```python
import withoutbg
from pathlib import Path

# Process product catalog
product_images = Path("products").glob("*.jpg")
results = withoutbg.remove_background_batch(
    list(product_images),
    output_dir="catalog-withoutbg/",
    api_key="sk_your_key"  # Use for best quality
)
```

### Social Media Automation
```python
import withoutbg
from PIL import Image

# Remove background and add custom background
foreground = withoutbg.remove_background("selfie.jpg", api_key="sk_key")
background = Image.open("gradient_bg.jpg")

# Composite images
background.paste(foreground, (0, 0), foreground)
background.save("social_post.jpg")
```

## 🔑 API Key Setup

1. **Get API Key**: Visit [withoutbg.com](https://withoutbg.com) to get your API key
2. **Set Environment Variable**:
   ```bash
   export WITHOUTBG_API_KEY="sk_your_api_key"
   ```
3. **Or pass directly in code**:
   ```python
   result = withoutbg.remove_background("image.jpg", api_key="sk_your_key")
   ```

## 🏗️ For Developers

### Local Development
```bash
# Clone repository
git clone https://github.com/withoutbg/withoutbg.git
cd withoutbg

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking  
mypy src/
```

## 📊 Usage Analytics

Track your API usage:

```python
from withoutbg.api import StudioAPI

api = StudioAPI(api_key="sk_your_key")
usage = api.get_usage()
print(usage)
```

## Commercial 


### API (Pay-per-use)
- ✅ Best quality processing
- ✅ Best quality results
- ✅ 99.9% uptime SLA
- ✅ Scalable infrastructure

[Try API →](https://withoutbg.com/remove-background)

## 📚 Documentation

- **[API Reference](https://withoutbg.com/documentation)** - Complete API documentation

## 🐛 Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/withoutbg/withoutbg/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/withoutbg/withoutbg/discussions)
- **Commercial Support**: [contact@withoutbg.com](mailto:contact@withoutbg.com)

## 🤗 Hugging Face

Find our models on Hugging Face:
- **[withoutbg/snap](https://huggingface.co/withoutbg/snap)** - Open source Snap model


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Third-Party Components
- **Depth Anything**: Apache 2.0 License

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for complete attribution.

## 🌟 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=withoutbg/withoutbg&type=Date)](https://star-history.com/#withoutbg/withoutbg&Date)

---

**[🎯 Get best quality results with withoutbg.com](https://withoutbg.com)**