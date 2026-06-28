# PyTorch  all topics 

## 1. Introduction to PyTorch

-   What is PyTorch?
-   Why PyTorch?
-   PyTorch vs TensorFlow
-   Dynamic Computation Graph
-   PyTorch Ecosystem
-   Installation and Setup

## 2. Tensors

-   Creating Tensors
-   Tensor Data Types
-   Tensor Shapes
-   Tensor Memory Layout
-   Tensor Operations
-   Indexing and Slicing
-   Reshaping
-   Broadcasting
-   Matrix Operations
-   Tensor Utilities

## 3. CUDA and GPU Computing

-   CPU vs GPU
-   CUDA Basics
-   Moving Tensors to GPU
-   Mixed Precision
-   Device Management

## 4. Automatic Differentiation (Autograd)

-   Computational Graph
-   requires_grad
-   Backpropagation
-   backward()
-   Gradient Tracking
-   Detach
-   torch.no_grad()

## 5. Building Neural Networks

-   nn.Module
-   Layers
-   Activation Functions
-   Forward Pass
-   Custom Modules
-   Model Summary

## 6. Loss Functions

-   Regression Losses
-   Classification Losses
-   Custom Loss Functions

## 7. Optimizers

-   SGD
-   Adam
-   AdamW
-   Learning Rate
-   Weight Decay
-   Momentum

## 8. Datasets and DataLoaders

-   Dataset Class
-   DataLoader
-   Batching
-   Shuffling
-   Custom Dataset
-   Collate Function

## 9. Training Loop

-   Forward Pass
-   Loss Calculation
-   Backward Pass
-   Optimizer Step
-   Zero Gradients

## 10. Validation and Testing

-   Evaluation Mode
-   Metrics
-   Validation Loop
-   Testing Loop

## 11. Saving and Loading Models

-   state_dict
-   torch.save()
-   torch.load()
-   Saving Checkpoints

## 12. Common PyTorch Utilities

-   Tensor Utilities
-   Random Seeds
-   Device Utilities
-   Model Utilities

## 13. Learning Rate Scheduling

-   StepLR
-   CosineAnnealingLR
-   ReduceLROnPlateau
-   Warmup Schedulers

## 14. Transfer Learning

-   Pretrained Models
-   Freezing Layers
-   Fine-Tuning
-   Feature Extraction

## 15. Convolutional Neural Networks (CNNs)

-   Convolution Layers
-   Pooling Layers
-   CNN Architectures

## 16. Recurrent Neural Networks (RNNs)

-   RNN
-   LSTM
-   GRU
-   Sequence Models

## 17. Attention Mechanisms

-   Attention Basics
-   Self-Attention
-   Multi-Head Attention

## 18. Transformer Models

-   Transformer Architecture
-   Encoder
-   Decoder
-   Positional Encoding

## 19. Distributed and Efficient Training

-   Data Parallelism
-   Distributed Data Parallel
-   Gradient Accumulation
-   Gradient Checkpointing
-   Mixed Precision Training

## 20. Model Quantization

-   Dynamic Quantization
-   Static Quantization
-   Quantization Aware Training

## 21. Model Export

-   TorchScript
-   ONNX Export
-   Deployment Basics

## 22. PyTorch Ecosystem

-   torchvision
-   torchaudio
-   torchtext
-   torchmetrics

## 23. Debugging and Profiling

-   Debugging Models
-   Profiling
-   Memory Optimization
-   Performance Tuning

## 24. Best Practices

-   Project Structure
-   Reproducibility
-   Experiment Tracking
-   Coding Standards

## 25. Capstone Projects

-   Image Classification
-   Object Detection
-   Text Classification
-   Language Modeling
-   Transformer Fine-Tuning
-   Custom AI Applications


> **Module 1 — Introduction to PyTorch**
> 
# Table of Contents
- What is PyTorch?
- Why was PyTorch created?
- Why do we need Deep Learning Frameworks?
- What problems does PyTorch solve?
- PyTorch vs TensorFlow
- PyTorch Ecosystem
- Installing PyTorch
- Creating Your First PyTorch Project
- Verifying Installation
- GPU Support
- Recommended IDEs
- Folder Structure
- First PyTorch Program
- Summary

---

# Before PyTorch Existed...

Imagine you are working as an AI engineer in **2015**.

You want to create a neural network that can recognize handwritten digits.

The mathematical equation looks something like this:

\[
Y = W \cdot X + b
\]

where

- **X** = Input
- **W** = Weights
- **b** = Bias
- **Y** = Prediction

Now imagine your neural network has:

- 5 Layers
- 10 Million Parameters

After every prediction, you must manually calculate

- Gradients
- Partial derivatives
- Chain Rule
- Weight updates

For millions of parameters...

Every iteration...

For every training sample...

This quickly becomes impossible.

Researchers spent more time writing mathematical derivatives than building better models.

---

# The Need for Deep Learning Frameworks

Deep Learning Frameworks automate the difficult parts of neural network training.

Instead of calculating gradients manually, frameworks automatically compute them using **Automatic Differentiation (Autograd)**.

A framework handles:

- Tensor operations
- GPU computation
- Gradient calculation
- Neural network layers
- Optimizers
- Model saving/loading
- Training loops
- Distributed training

This lets developers focus on designing models instead of implementing calculus from scratch.

---

# What is PyTorch?

PyTorch is an **open-source Deep Learning Framework** developed by **Facebook AI Research (FAIR)**.

It provides tools for building, training, and deploying machine learning and deep learning models efficiently.

At its core, PyTorch is built around **tensors** and **automatic differentiation**, making it easy to implement everything from simple neural networks to state-of-the-art large language models.

---

# Official Definition

> PyTorch is an open-source machine learning framework based on the Torch library, designed for applications such as computer vision, natural language processing, reinforcement learning, and large-scale deep learning research.

---

# Why is PyTorch So Popular?

PyTorch became popular because it combines flexibility with performance.

Some key reasons include:

- Python-friendly API
- Dynamic computation graphs
- Easy debugging
- GPU acceleration
- Large research community
- Strong industry adoption
- Extensive ecosystem

Today, PyTorch powers many modern AI systems, including models developed by organizations such as Meta, OpenAI, Microsoft, and Hugging Face.

---

# Where is PyTorch Used?

PyTorch is used across many AI domains.

| Domain | Example |
|---------|----------|
| Computer Vision | Image Classification |
| NLP | Chatbots |
| LLMs | Llama, Qwen, Gemma |
| Speech | Speech Recognition |
| Reinforcement Learning | Robotics |
| Medical AI | Disease Prediction |
| Recommendation Systems | Netflix, Amazon |

---

# Why Not Use Only NumPy?

NumPy is excellent for numerical computation, but it lacks several features required for deep learning.

For example, NumPy does not provide:

- Automatic gradient computation
- Neural network modules
- GPU support by default
- Optimizers
- Training utilities
- Pretrained deep learning models

PyTorch extends numerical computation with these deep learning capabilities.

---

# NumPy vs PyTorch

| Feature | NumPy | PyTorch |
|----------|--------|----------|
| Arrays | ✅ | ✅ |
| GPU Support | ❌ | ✅ |
| Autograd | ❌ | ✅ |
| Neural Networks | ❌ | ✅ |
| Optimizers | ❌ | ✅ |
| Distributed Training | ❌ | ✅ |

---

# PyTorch vs TensorFlow

| Feature | PyTorch | TensorFlow |
|----------|----------|------------|
| Learning Curve | Easy | Moderate |
| Debugging | Easy | More Complex |
| Python Integration | Excellent | Good |
| Research Popularity | Very High | High |
| Production Deployment | Excellent | Excellent |

Both frameworks are widely used. PyTorch is especially popular in research and modern LLM development due to its intuitive design.

---

# PyTorch Ecosystem

PyTorch is more than a single library.

```text
                 PyTorch
                     │
 ┌────────────┬────────────┬────────────┬────────────┐
 │            │            │            │
torchvision torchaudio torchtext torchdata
 │            │            │            │
Images      Audio        NLP        Data Loading
```

Some commonly used libraries include:

- **torch** – Core tensor and neural network functionality
- **torchvision** – Computer vision datasets and models
- **torchaudio** – Audio processing
- **torchtext** – NLP utilities
- **torchmetrics** – Evaluation metrics

As you progress, you'll also use:

- transformers
- datasets
- accelerate
- peft
- trl
- unsloth

These libraries build on top of PyTorch for LLM development.

---

# Installing PyTorch

Always refer to the official installation page:

https://pytorch.org/get-started/locally/

At the time of writing, a common installation command is:

### CPU Version

```bash
pip install torch torchvision torchaudio
```

### CUDA Version (Example)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **Note:** The CUDA version depends on your installed NVIDIA drivers and CUDA runtime. Always use the command recommended by the official website for your system.

---

# Create a Virtual Environment

Using a virtual environment keeps project dependencies isolated.

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

# Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# Install PyTorch

```bash
pip install torch torchvision torchaudio
```

---

# Verify Installation

Create a file named:

```text
test.py
```

Add:

```python
import torch

print(torch.__version__)
```

Run:

```bash
python test.py
```

Example output:

```text
2.8.0
```

Your version may differ.

---

# Check GPU Availability

```python
import torch

print(torch.cuda.is_available())
```

Possible outputs:

```text
True
```

or

```text
False
```

If `False`, PyTorch will use the CPU.

---

# Get GPU Name

```python
import torch

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Example output:

```text
NVIDIA GeForce RTX 4060
```

---

# Check Current Device

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
```

Output:

```text
cuda
```

or

```text
cpu
```

This pattern will appear in almost every PyTorch project.

---

# Recommended Development Environment

For most learners:

- Visual Studio Code
- Python Extension
- Jupyter Extension (optional)
- Git
- GitHub

Useful extensions:

- Python
- Pylance
- Jupyter
- Error Lens
- GitLens

---

# Recommended Project Structure

```text
PyTorch-Projects/
│
├── data/
├── notebooks/
├── models/
├── src/
├── checkpoints/
├── outputs/
├── requirements.txt
├── README.md
└── train.py
```

Keeping a clean project structure becomes increasingly important as projects grow.

---

# Your First PyTorch Program

Create a file named:

```text
hello_pytorch.py
```

```python
import torch

print("Hello PyTorch!")
print(torch.__version__)
```

Run:

```bash
python hello_pytorch.py
```

Expected output:

```text
Hello PyTorch!
2.x.x
```

---

# What We'll Learn Next

Now that the environment is ready, the next module focuses on the most important building block in PyTorch:

> **Tensors**

Every operation in PyTorch—whether it's a neural network, image processing, or an LLM—starts with tensors.

Understanding tensors thoroughly is essential before moving on to neural networks, autograd, or transformers.

---

# Key Takeaways

- PyTorch is an open-source deep learning framework developed by Meta AI.
- It automates gradient computation through Automatic Differentiation.
- It supports both CPU and GPU execution.
- PyTorch provides an ecosystem for vision, NLP, audio, and large language models.
- A virtual environment helps isolate project dependencies.
- Verifying your installation and GPU setup is the first step before building models.
- Tensors are the foundation of all computations in PyTorch.

---
