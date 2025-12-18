# yndx-llm-scaling-week-2025

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

This project focuses on enhancing the scalability and efficiency of large language models through advanced optimization techniques. It prioritizes critical performance improvements while maintaining accuracy, enabling more effective deployment of LLMs for large-scale applications.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Core features

1. **SwiGLU Activation Function Optimization**: Implements a highly optimized Triton kernel for the SwiGLU activation function (C = A * sigmoid(A) * B) with autotuning for different block sizes to maximize GPU performance. Uses mixed precision calculations and memory masking for edge cases.
2. **Int8 Weight Quantization**: Provides per-row weight quantization to int8 format with dynamic scale calculation. Includes safety checks for numerical stability and ensures proper memory alignment for subsequent matrix operations.
3. **Quantized Matrix Multiplication (W8A8)**: Implements a custom Triton kernel for matrix multiplication using int8 quantized weights and activations. Supports split-k parallelism and includes dequantization logic to produce float16 outputs while maintaining numerical accuracy.

---

## Installation

Install yndx-llm-scaling-week-2025 using one of the following methods:

**Build from source:**

1. Clone the yndx-llm-scaling-week-2025 repository:
```sh
git clone https://github.com/DRMPN/yndx-llm-scaling-week-2025
```

2. Navigate to the project directory:
```sh
cd yndx-llm-scaling-week-2025
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Contributing

- **[Report Issues](https://github.com/DRMPN/yndx-llm-scaling-week-2025/issues)**: Submit bugs found or log feature requests for the project.

---

## Citation

If you use this software, please cite it as below.

### APA format

```text
DRMPN (2025). yndx-llm-scaling-week-2025 repository [Computer software]. https://github.com/DRMPN/yndx-llm-scaling-week-2025
```

### BibTeX format

```bibtex
@misc{yndx-llm-scaling-week-2025,
    author = {DRMPN},
    title = {yndx-llm-scaling-week-2025 repository},
    year = {2025},
    publisher = {github.com},
    journal = {github.com repository},
    howpublished = {\url{https://github.com/DRMPN/yndx-llm-scaling-week-2025.git}},
    url = {https://github.com/DRMPN/yndx-llm-scaling-week-2025.git}
}
```

---