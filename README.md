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

> **Module 2 — Tensors (Part 1: Fundamentals)**

# 📚 Table of Contents

- The Story Behind Tensors
- Why Do We Need Tensors?
- What is a Tensor?
- Tensor vs NumPy Array
- Scalars, Vectors, Matrices and Tensors
- Tensor Dimensions (Rank)
- Tensor Shape
- Number of Elements
- Creating Your First Tensor
- Tensor Data Types
- Tensor Devices (CPU & GPU)
- Tensor Properties
- Summary
- Practice Questions

---

# 📖 The Story Behind Tensors

Before learning PyTorch, let's understand **why tensors exist**.

Imagine you are building an AI model that recognizes cats in images.

A computer doesn't understand images like humans do.

When you open a picture like this:

```
🐱
```

You see a cat.

But the computer actually sees something like this:

```
[[255, 120, 35],
 [100,  90, 10],
 [220, 150, 80],
 ...
]
```

These are **numbers** representing pixel values.

Similarly,

A sentence like

```
I love AI
```

becomes

```
[105, 2034, 501]
```

Audio becomes numbers.

Videos become numbers.

Medical scans become numbers.

Everything inside Deep Learning eventually becomes **numbers**.

The question is...

**How do we efficiently store and manipulate millions (or even billions) of numbers?**

The answer is:

> **Tensor**

---

# Why Do We Need Tensors?

Suppose you want to train an image classifier.

Each image is

```
224 × 224 × 3
```

That means

```
224 × 224 × 3
=
150,528 numbers
```

If you train on

```
1000 images
```

you already have

```
150 Million+
numbers
```

Performing operations on such large datasets using Python lists would be extremely slow.

Tensors are optimized data structures that allow fast numerical computation, parallel execution, and GPU acceleration.

---

# What is a Tensor?

A **Tensor** is the fundamental data structure in PyTorch.

> **Definition:**  
> A tensor is a multi-dimensional array capable of storing numerical data and performing high-performance mathematical operations on CPUs and GPUs.

Think of tensors as the "containers" that hold data for machine learning models.

Everything in PyTorch—inputs, outputs, weights, gradients, and predictions—is represented as tensors.

---

# Real-Life Analogy

Imagine different ways of storing information:

| Data | Representation |
|-------|----------------|
| Single temperature | Scalar |
| List of marks | Vector |
| Excel sheet | Matrix |
| RGB Image | Tensor |
| Video | Tensor |
| LLM Training Data | Tensor |

As the complexity of data increases, we move from simple numbers to higher-dimensional tensors.

---

# Scalars, Vectors, Matrices and Tensors

## Scalar (0D Tensor)

A single value.

Example:

```
7
```

Temperature

```
32°C
```

Age

```
21
```

Python

```python
import torch

scalar = torch.tensor(7)

print(scalar)
```

Output

```
tensor(7)
```

---

## Vector (1D Tensor)

A collection of numbers.

Example

```
[10,20,30,40]
```

Python

```python
vector = torch.tensor([10,20,30,40])

print(vector)
```

Output

```
tensor([10,20,30,40])
```

---

## Matrix (2D Tensor)

Rows and Columns.

Example

```
[
 [1,2,3],
 [4,5,6]
]
```

Python

```python
matrix = torch.tensor([
    [1,2,3],
    [4,5,6]
])

print(matrix)
```

Output

```
tensor([[1,2,3],
        [4,5,6]])
```

---

## 3D Tensor

Imagine stacking multiple matrices.

```
[
 Matrix1

 Matrix2

 Matrix3
]
```

Example:

```
RGB Image

Height

Width

Channels
```

---

## 4D Tensor

Most Deep Learning models use 4D tensors.

Example

```
Batch Size

Height

Width

Channels
```

or

```
Batch

Channel

Height

Width
```

---

# Visual Representation

```
Scalar

7
```

↓

```
Vector

[1 2 3]
```

↓

```
Matrix

1 2 3

4 5 6
```

↓

```
Tensor

Matrix

Matrix

Matrix
```

---

# Tensor Dimensions (Rank)

Dimension tells us **how many axes** a tensor has.

| Tensor | Rank |
|----------|------|
| Scalar | 0 |
| Vector | 1 |
| Matrix | 2 |
| Image | 3 |
| Batch of Images | 4 |

Example

```python
scalar = torch.tensor(5)

print(scalar.ndim)
```

Output

```
0
```

---

Vector

```python
vector = torch.tensor([1,2,3])

print(vector.ndim)
```

Output

```
1
```

---

Matrix

```python
matrix = torch.tensor([
    [1,2],
    [3,4]
])

print(matrix.ndim)
```

Output

```
2
```

---

# Tensor Shape

Shape tells us

> **How many elements exist along each dimension.**

Example

```python
matrix = torch.tensor([
    [1,2,3],
    [4,5,6]
])
```

Shape

```
(2,3)
```

Meaning

```
2 Rows

3 Columns
```

Python

```python
print(matrix.shape)
```

Output

```
torch.Size([2,3])
```

---

Another Example

```python
image = torch.rand(3,224,224)

print(image.shape)
```

Output

```
torch.Size([3,224,224])
```

Meaning

```
Channels

Height

Width
```

---

# Number of Elements

Sometimes we need to know how many values exist inside a tensor.

PyTorch provides

```python
numel()
```

Example

```python
matrix = torch.tensor([
    [1,2,3],
    [4,5,6]
])

print(matrix.numel())
```

Output

```
6
```

Because

```
2 × 3 = 6
```

---

# Creating Your First Tensor

The simplest way to create a tensor is using

```python
torch.tensor()
```

Example

```python
import torch

numbers = torch.tensor([10,20,30])

print(numbers)
```

Output

```
tensor([10,20,30])
```

---

Matrix Example

```python
marks = torch.tensor([
    [80,90],
    [70,95]
])

print(marks)
```

Output

```
tensor([[80,90],
        [70,95]])
```

---

# Tensor Data Types

Every tensor stores data of a specific type.

Common data types:

| Data Type | Description |
|------------|-------------|
| torch.int8 | 8-bit Integer |
| torch.int32 | 32-bit Integer |
| torch.int64 | 64-bit Integer |
| torch.float16 | Half Precision Float |
| torch.float32 | Standard Float |
| torch.float64 | Double Precision Float |
| torch.bool | Boolean |

Check the datatype

```python
x = torch.tensor([1,2,3])

print(x.dtype)
```

Output

```
torch.int64
```

---

# Tensor Device

Every tensor lives on a device.

Possible devices:

- CPU
- GPU

Check device

```python
x = torch.tensor([1,2,3])

print(x.device)
```

Output

```
cpu
```

If moved to GPU

```
cuda:0
```

---

# Tensor Properties

Useful properties:

```python
x = torch.rand(3,4)
```

Shape

```python
x.shape
```

Dimensions

```python
x.ndim
```

Datatype

```python
x.dtype
```

Device

```python
x.device
```

Number of Elements

```python
x.numel()
```

Size

```python
x.size()
```

These properties are frequently used when debugging models.

---

# Summary

In this chapter, you learned:

- Why tensors are essential in deep learning.
- The difference between scalars, vectors, matrices, and higher-dimensional tensors.
- How tensor dimensions (rank) describe the number of axes.
- How tensor shapes specify the size of each dimension.
- How to create tensors using `torch.tensor()`.
- How to inspect tensor properties such as shape, data type, device, and number of elements.

Tensors are the foundation of every PyTorch program. Before building neural networks, it's important to become comfortable creating and inspecting them.

---

# 📝 Practice Questions

1. What is a tensor?
2. Why are tensors preferred over Python lists for deep learning?
3. Explain the difference between a scalar, vector, matrix, and tensor.
4. What is the rank (dimension) of a tensor?
5. What is the difference between `shape` and `numel()`?
6. How do you create a tensor in PyTorch?
7. How do you check the datatype of a tensor?
8. How do you check whether a tensor is on the CPU or GPU?
9. What does `ndim` return?
10. What does `torch.Size([3, 224, 224])` represent?

---

# 🚀 What's Next?

In **Module 2 – Part 2**, you'll learn how to create tensors using PyTorch factory functions, including:

- `torch.zeros()`
- `torch.ones()`
- `torch.empty()`
- `torch.rand()`
- `torch.randn()`
- `torch.randint()`
- `torch.arange()`
- `torch.linspace()`
- `torch.eye()`
- `torch.full()`

You'll also understand when to use each function in real-world machine learning and LLM workflows.


> # Module 3 – CUDA & GPU Computing (Part 1)
>
> **Topic:** CPU vs GPU & CUDA Fundamentals

# 📚 Table of Contents

- The Story Behind GPU Computing
- Why CPU Becomes Slow
- What is a CPU?
- What is a GPU?
- CPU vs GPU
- Parallel Computing
- Why Deep Learning Needs GPUs
- What is CUDA?
- CUDA Architecture
- CUDA Cores
- GPU Memory (VRAM)
- CUDA Versions
- NVIDIA Driver
- CUDA Toolkit
- cuDNN
- How PyTorch Uses CUDA
- Summary
- Interview Questions
- Practice Questions

---

# 📖 The Story Behind GPU Computing

Let's travel back a few years.

Imagine you are training a simple neural network.

The dataset contains only

```
10,000 Images
```

Each image is

```
224 × 224 × 3
```

Your neural network has

```
15 Million Parameters
```

Every epoch requires

```
Forward Pass

↓

Loss Calculation

↓

Backward Pass

↓

Gradient Computation

↓

Weight Update
```

Now imagine repeating this process

```
100 Times
```

Your computer starts slowing down.

Training takes

```
Hours...

Sometimes Days...

Sometimes Weeks...
```

The problem wasn't the neural network.

The problem was the hardware.

Researchers realized that CPUs were never designed for performing billions of mathematical operations simultaneously.

A different type of processor was needed.

That processor was the **GPU**.

---

# Why CPUs Become Slow

A CPU is designed to perform many different kinds of tasks.

Examples:

- Opening applications
- Running operating systems
- Browsing the internet
- Playing music
- Managing files

Because of this, CPUs focus on being **versatile**, not on performing the same mathematical operation millions of times.

Deep learning, however, repeatedly performs operations such as:

```
Matrix Multiplication

Addition

Multiplication

Activation Functions

Gradient Calculations
```

These operations are highly repetitive and can be executed in parallel.

---

# What is a CPU?

CPU stands for

> **Central Processing Unit**

It is the brain of your computer.

A modern CPU typically has

```
4

8

12

16

24

32 Cores
```

Each core is extremely powerful and optimized for sequential execution.

Example

```
Task 1

↓

Task 2

↓

Task 3

↓

Task 4
```

One core completes one task before moving to the next.

This makes CPUs excellent for general-purpose computing.

---

# What is a GPU?

GPU stands for

> **Graphics Processing Unit**

Originally, GPUs were designed to render graphics for games.

Rendering millions of pixels requires performing identical mathematical operations simultaneously.

Manufacturers realized that this architecture is also perfect for deep learning.

Instead of a few powerful cores, GPUs contain **thousands of smaller cores** capable of executing many operations at once.

---

# CPU vs GPU

Imagine you need to solve

```
10,000 Math Problems
```

### CPU

You hire

```
8 Expert Teachers
```

Each teacher solves one problem at a time.

```
Teacher 1 → Problem 1

Teacher 2 → Problem 2

Teacher 3 → Problem 3
...
```

The teachers are very smart but there are only a few of them.

---

### GPU

Now imagine hiring

```
10,000 Students
```

Each student solves one simple problem.

```
Student 1 → Problem 1

Student 2 → Problem 2

Student 3 → Problem 3

...

Student 10000 → Problem 10000
```

Each student is slower than a teacher,

but because thousands work simultaneously,

the entire task finishes much faster.

This is exactly how GPUs accelerate deep learning.

---

# CPU vs GPU Comparison

| Feature | CPU | GPU |
|----------|-----|-----|
| Full Form | Central Processing Unit | Graphics Processing Unit |
| Number of Cores | Few (4–32) | Thousands |
| Optimized For | General Computing | Parallel Computation |
| Best At | Sequential Tasks | Matrix Operations |
| Deep Learning | Slow | Extremely Fast |
| Cost per Computation | Higher | Lower |

---

# Sequential vs Parallel Computing

### Sequential Processing

```
Task 1

↓

Task 2

↓

Task 3

↓

Task 4
```

One task finishes before the next begins.

---

### Parallel Processing

```
Task 1

Task 2

Task 3

Task 4

↓

All execute together
```

Deep learning relies heavily on parallel processing because matrix operations can be divided across many cores.

---

# Why Neural Networks Love GPUs

Consider multiplying two matrices:

```
1024 × 1024

×

1024 × 1024
```

This requires **millions of multiply-add operations**.

A CPU processes these operations using only a limited number of cores.

A GPU distributes the work across thousands of CUDA cores, allowing the computation to complete much faster.

The larger the model, the greater the performance benefit.

---

# What is CUDA?

CUDA stands for

> **Compute Unified Device Architecture**

CUDA is a platform developed by **NVIDIA** that allows programmers to use NVIDIA GPUs for general-purpose computation.

Without CUDA,

your GPU would mainly be used for graphics.

With CUDA,

the GPU can perform machine learning computations.

PyTorch communicates with CUDA to execute tensor operations on NVIDIA GPUs.

---

# CUDA Architecture

```
                    PyTorch

                       │

                 CUDA Runtime

                       │

                NVIDIA Driver

                       │

                 NVIDIA GPU
```

When you write:

```python
tensor = tensor.to("cuda")
```

PyTorch sends the tensor to the CUDA runtime, which communicates with the NVIDIA driver and executes the computation on the GPU.

---

# CUDA Cores

CUDA cores are the small processing units inside an NVIDIA GPU.

Example:

| GPU | CUDA Cores |
|------|------------|
| RTX 3050 | 2560 |
| RTX 3060 | 3584 |
| RTX 4060 | 3072 |
| RTX 4090 | 16384 |

More CUDA cores generally allow more parallel computations, though memory bandwidth and architecture also matter.

---

# GPU Memory (VRAM)

Unlike CPUs, GPUs have their own dedicated memory called **VRAM**.

Examples:

| GPU | VRAM |
|------|------|
| RTX 3050 | 8 GB |
| RTX 3060 | 12 GB |
| RTX 4060 | 8 GB |
| RTX 4090 | 24 GB |

During training, VRAM stores:

- Model parameters
- Input tensors
- Gradients
- Optimizer states
- Intermediate activations

If the required memory exceeds available VRAM, you'll encounter an **Out of Memory (OOM)** error.

---

# NVIDIA Driver

The NVIDIA driver acts as the bridge between the operating system and the GPU.

Without the correct driver, PyTorch cannot communicate with the GPU.

---

# CUDA Toolkit

The CUDA Toolkit includes:

- CUDA Compiler (`nvcc`)
- Runtime Libraries
- Development Tools
- Debugging Utilities

It provides the software environment needed to build and run CUDA applications.

---

# What is cuDNN?

cuDNN stands for

> **CUDA Deep Neural Network Library**

It is a highly optimized library for deep learning operations such as:

- Convolutions
- Pooling
- Activation Functions
- Batch Normalization
- RNN Operations

Frameworks like PyTorch and TensorFlow use cuDNN internally to speed up neural network training.

---

# How PyTorch Uses CUDA

When you write:

```python
import torch

x = torch.tensor([1, 2, 3], device="cuda")
```

PyTorch:

1. Creates the tensor.
2. Allocates memory on the GPU.
3. Transfers the data.
4. Executes future operations on the GPU instead of the CPU.

You don't need to write CUDA code yourself—PyTorch handles the interaction.

---

# Key Takeaways

- CPUs are optimized for general-purpose, sequential computing.
- GPUs are optimized for parallel numerical computations.
- Deep learning relies heavily on matrix operations, making GPUs ideal.
- CUDA enables NVIDIA GPUs to perform general-purpose computation.
- PyTorch uses CUDA to execute tensor operations on the GPU.
- VRAM stores tensors, gradients, activations, and model parameters during training.
- cuDNN provides optimized implementations of common deep learning operations.

---

# 📝 Interview Questions

1. Why are GPUs faster than CPUs for deep learning?
2. What is CUDA?
3. What is the role of the CUDA Toolkit?
4. What is cuDNN?
5. What is VRAM, and why is it important?
6. Can PyTorch use a GPU without CUDA?
7. Explain sequential vs parallel computing.
8. Why do large language models require GPUs?

---

# 💻 Practice Questions

1. Explain the difference between CPU and GPU in your own words.
2. Why is matrix multiplication considered a parallel operation?
3. What components are required for PyTorch to use an NVIDIA GPU?
4. Research your GPU model and note its CUDA cores and VRAM.
5. What happens if a model requires more VRAM than is available?

---

> # Module 3 – CUDA & GPU Computing (Part 2)
>
> **Topic:** Installing CUDA & Verifying GPU Support
>

# 📚 Table of Contents

- Introduction
- Understanding the Software Stack
- Step 1 – Check Your GPU
- Step 2 – Install NVIDIA Driver
- Step 3 – Install PyTorch
- Step 4 – Verify Installation
- Checking CUDA Availability
- Checking GPU Information
- Understanding CUDA Versions
- Common Commands
- Troubleshooting
- Best Practices
- Summary
- Interview Questions
- Exercises

---

# 📖 Introduction

In the previous chapter, we learned that GPUs make Deep Learning significantly faster.

But simply owning an NVIDIA GPU isn't enough.

Your computer needs several software components that work together before PyTorch can use the GPU.

Think of it like driving a car.

Having the car alone isn't enough.

You also need:

- Fuel
- Engine
- Driver
- Roads

Similarly, PyTorch needs several components before it can communicate with your GPU.

---

# Understanding the Software Stack

When you execute

```python
tensor = tensor.to("cuda")
```

many things happen behind the scenes.

```
Your Python Code
        │
        ▼
     PyTorch
        │
        ▼
 CUDA Runtime Libraries
        │
        ▼
 NVIDIA Driver
        │
        ▼
 NVIDIA GPU
```

Every layer has an important responsibility.

If even one layer is missing,

PyTorch cannot use the GPU.

---

# What Do We Need?

To use GPU acceleration, we generally need:

✅ NVIDIA GPU

↓

✅ NVIDIA Driver

↓

✅ PyTorch (CUDA Build)

Notice something interesting.

You **do not always need to install the full CUDA Toolkit** just to train models.

Modern PyTorch wheels already include the CUDA runtime libraries they require.

---

# Step 1 — Check Your GPU

Before installing anything, verify that your computer actually has an NVIDIA GPU.

### Windows

Open

```
Task Manager

↓

Performance

↓

GPU
```

Example

```
GPU 0

NVIDIA RTX 4060
```

---

Another method

Open Command Prompt

```bash
nvidia-smi
```

If installed correctly, you'll see something like

```
+----------------------------------+

NVIDIA-SMI 580.xx

Driver Version: 580.xx

CUDA Version: 13.x

GPU Memory

Processes

+----------------------------------+
```

---

If you receive

```
'nvidia-smi' is not recognized
```

then either

- NVIDIA driver is missing
- Driver installation failed
- PATH issue

---

# Step 2 — Install the NVIDIA Driver

The driver is the bridge between your operating system and the GPU.

Without it,

PyTorch cannot communicate with the hardware.

Always download drivers from the official NVIDIA website:

```
https://www.nvidia.com/download/
```

Select:

- GPU Series
- GPU Model
- Operating System

Download

Install

Restart your computer.

---

# Verify Driver Installation

Open Command Prompt

```bash
nvidia-smi
```

Expected output

```
GPU Name

Driver Version

CUDA Version

Memory Usage

Temperature

Power Usage
```

This confirms your operating system can communicate with the GPU.

---

# Step 3 — Install PyTorch

Visit

```
https://pytorch.org/get-started/locally/
```

Choose

```
Operating System

Package Manager

Python

CUDA Version
```

The website automatically generates the correct installation command.

Example

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

This installs the CUDA-enabled build of PyTorch.

---

# Why Don't We Install CUDA Toolkit First?

Many beginners think

```
CUDA Toolkit

↓

PyTorch
```

is mandatory.

In reality,

most users only need

```
NVIDIA Driver

↓

CUDA-enabled PyTorch
```

PyTorch ships with the CUDA runtime it needs.

You only need the full CUDA Toolkit when:

- Writing CUDA C++ code
- Building custom CUDA extensions
- Developing GPU kernels

---

# Step 4 — Verify Installation

Create

```
check_gpu.py
```

```python
import torch

print(torch.__version__)
```

Output

```
2.x.x
```

---

# Is CUDA Available?

```python
import torch

print(torch.cuda.is_available())
```

Output

```
True
```

means

PyTorch successfully detected your GPU.

If

```
False
```

then PyTorch is using the CPU.

---

# How Many GPUs?

```python
import torch

print(torch.cuda.device_count())
```

Example

```
1
```

If your workstation has multiple GPUs,

you may see

```
4
```

or

```
8
```

---

# Current GPU

```python
import torch

print(torch.cuda.current_device())
```

Output

```
0
```

GPU numbering starts from

```
0
```

---

# GPU Name

```python
import torch

print(torch.cuda.get_device_name(0))
```

Example

```
NVIDIA GeForce RTX 4060
```

---

# Complete Example

```python
import torch

print("PyTorch Version :", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())
print("GPU Count :", torch.cuda.device_count())

if torch.cuda.is_available():

    print("Current GPU :", torch.cuda.current_device())

    print("GPU Name :", torch.cuda.get_device_name(0))
```

Example Output

```
PyTorch Version : 2.8.0

CUDA Available : True

GPU Count : 1

Current GPU : 0

GPU Name : NVIDIA RTX 4060
```

---

# Understanding CUDA Versions

Many beginners confuse these three versions.

There are actually three different version numbers.

```
NVIDIA Driver Version

↓

CUDA Runtime Version

↓

PyTorch CUDA Build
```

Example

```
Driver

580.xx
```

```
CUDA Runtime

13.0
```

```
PyTorch

cu128
```

These are related,

but they are **not the same thing**.

---

# Check CUDA Version Used by PyTorch

```python
import torch

print(torch.version.cuda)
```

Example

```
12.8
```

This shows the CUDA version against which PyTorch was built.

---

# Check cuDNN Version

```python
import torch

print(torch.backends.cudnn.version())
```

Example

```
9200
```

This confirms cuDNN is available.

---

# Device Object

Instead of writing

```python
"cuda"
```

everywhere,

PyTorch recommends using

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

Now

```python
print(device)
```

Output

```
cuda
```

or

```
cpu
```

This makes your code portable across systems with and without GPUs.

---

# Visual Workflow

```
Install Driver

↓

Install PyTorch

↓

Import torch

↓

Check CUDA

↓

Detect GPU

↓

Train Models
```

---

# Common Commands

Check GPU

```python
torch.cuda.is_available()
```

GPU Count

```python
torch.cuda.device_count()
```

Current GPU

```python
torch.cuda.current_device()
```

GPU Name

```python
torch.cuda.get_device_name(0)
```

CUDA Version

```python
torch.version.cuda
```

cuDNN Version

```python
torch.backends.cudnn.version()
```

---

# Common Problems

## Problem 1

```
CUDA Available : False
```

Possible Reasons

- NVIDIA driver missing
- Wrong PyTorch build
- Unsupported GPU
- Driver too old

---

## Problem 2

```
Torch not compiled with CUDA enabled
```

Reason

You installed the CPU-only version of PyTorch.

Solution

Install the CUDA-enabled wheel from the official PyTorch website.

---

## Problem 3

```
nvidia-smi not found
```

Reason

Driver isn't installed correctly.

---

## Problem 4

```
No NVIDIA GPU detected
```

Possible reasons

- Laptop using integrated graphics
- Driver issue
- GPU disabled in BIOS
- Hardware not present

---

# Best Practices

✅ Always install drivers from NVIDIA.

✅ Always copy the installation command from the official PyTorch website.

✅ Use virtual environments for projects.

✅ Check `torch.cuda.is_available()` before training.

✅ Write device-independent code using `torch.device()`.

---

# Summary

In this chapter, you learned:

- How the GPU software stack works.
- Why the NVIDIA driver is required.
- How to install a CUDA-enabled version of PyTorch.
- How to verify GPU support.
- How to inspect CUDA, cuDNN, and GPU information.
- Common installation problems and how to troubleshoot them.

At this point, your system is ready to execute PyTorch programs on the GPU.

---

# 🎯 Interview Questions

1. What is the role of the NVIDIA Driver?
2. What is `nvidia-smi`?
3. Does PyTorch always require the CUDA Toolkit?
4. What does `torch.cuda.is_available()` do?
5. What is the difference between Driver Version and CUDA Version?
6. How can you check the number of GPUs?
7. How do you write device-independent PyTorch code?
8. Why is `torch.device()` recommended?

---

# 📝 Practice Exercises

### Exercise 1

Print your GPU name.

---

### Exercise 2

Print the CUDA version used by PyTorch.

---

### Exercise 3

Print the cuDNN version.

---

### Exercise 4

Create a `device` object and print whether your program is using CPU or GPU.

---

### Exercise 5

Run `nvidia-smi` and identify:

- Driver Version
- CUDA Version
- GPU Memory
- GPU Name

---

# 🚀 What's Next?

In **Module 3 – Part 3**, we'll finally start using the GPU in code.

You'll learn:

- `tensor.to()`
- `.cuda()`
- `.cpu()`
- Moving entire models to GPU
- Common device mismatch errors
- Writing GPU-compatible training loops
- Performance tips and best practices

This is where your PyTorch programs begin taking advantage of GPU acceleration.


# Module 3 – CUDA & GPU Computing (Part 3)

> **Topic:** Moving Tensors & Models to GPU

**Goal**

By the end of this chapter you will learn

- Why tensors need to move to GPU
- What is `torch.device`
- How `.to()` works
- How `.cuda()` works
- How `.cpu()` works
- Moving complete models
- Device-independent programming
- Common CUDA errors
- Best Practices

---

# Story

Imagine you bought an expensive RTX 4090 GPU.

You installed

- CUDA
- PyTorch
- NVIDIA Drivers

Everything is ready.

Now you write

```python
import torch

x = torch.tensor([1,2,3])

print(x)
```

Output

```
tensor([1,2,3])
```

You think

> "Great! My GPU is working."

Actually...

**No.**

Your tensor is still stored inside **CPU RAM**.

Your GPU hasn't done any work yet.

Until you explicitly move a tensor to the GPU,

PyTorch continues using the CPU.

This is one of the biggest beginner mistakes.

---

# CPU Memory vs GPU Memory

Your computer has two different memories.

```
                Computer

          +-----------------+

          |     CPU RAM     |

          +-----------------+

                 ↑

          Stores Normal Data

                 ↓

          +-----------------+

          |    GPU VRAM     |

          +-----------------+

           Stores GPU Data
```

A tensor cannot exist in both places simultaneously.

It must be stored on

- CPU

OR

- GPU

---

# Why Move Tensors?

Suppose we have

```
Input Image

↓

Neural Network

↓

Prediction
```

If

Input is on CPU

and

Model is on GPU

PyTorch doesn't know how to perform operations.

Both must exist on the **same device**.

---

# Device

A **Device** tells PyTorch

> "Where should this tensor live?"

Possible devices

```
cpu
```

or

```
cuda
```

---

# Creating a Device Object

Instead of writing

```python
"cuda"
```

everywhere,

PyTorch recommends

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)
```

Now

```python
print(device)
```

Output

```
cuda
```

or

```
cpu
```

---

# Why Use torch.device()?

Imagine

Laptop A

```
RTX 4060
```

Laptop B

```
No GPU
```

If your code contains

```python
tensor.cuda()
```

Laptop B will crash.

Instead,

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)
```

works on both systems.

This is called

**Device Independent Programming**

---

# Moving Tensor to GPU

CPU Tensor

```python
import torch

x = torch.tensor([1,2,3])

print(x.device)
```

Output

```
cpu
```

Move

```python
x = x.to("cuda")
```

Now

```python
print(x.device)
```

Output

```
cuda:0
```

---

# Using Device Variable

Instead of

```python
x.to("cuda")
```

write

```python
x = x.to(device)
```

Advantages

- Portable
- Cleaner
- Recommended

---

# .to()

General Syntax

```python
tensor.to(device)
```

Example

```python
device = torch.device("cuda")

x = torch.tensor([1,2,3])

x = x.to(device)
```

---

# .cuda()

Shortcut

```python
x = x.cuda()
```

Same as

```python
x.to("cuda")
```

---

Example

```python
x = torch.tensor([10,20])

x = x.cuda()
```

Output Device

```
cuda:0
```

---

# .cpu()

Moves tensor back

```python
x = x.cpu()
```

Now

```
GPU

↓

CPU
```

Example

```python
x = x.cpu()
```

Output

```
cpu
```

---

# Complete Example

```python
import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

x = torch.tensor([1,2,3])

print(x.device)

x = x.to(device)

print(x.device)
```

Possible Output

```
cpu

cuda:0
```

---

# Creating Tensor Directly on GPU

Instead of

```python
x = torch.tensor([1,2,3])

x = x.to(device)
```

You can directly write

```python
x = torch.tensor(
    [1,2,3],
    device=device
)
```

Much faster.

---

# Multiple Tensors

```python
a = torch.rand(5,5).to(device)

b = torch.rand(5,5).to(device)

c = a + b
```

Everything happens on GPU.

---

# Checking Tensor Device

```python
print(a.device)
```

Output

```
cuda:0
```

---

# Moving Neural Network

Tensor is not enough.

The model also needs GPU.

Example

```python
model = MyModel()
```

Move

```python
model = model.to(device)
```

Now

Every layer

Every parameter

Every weight

moves to GPU.

---

# Complete Flow

```
Dataset

↓

Tensor

↓

Move Tensor

↓

Move Model

↓

Forward Pass

↓

Loss

↓

Backward Pass

↓

Optimizer
```

---

# Training Example

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

model = MyModel()

model = model.to(device)

for images, labels in train_loader:

    images = images.to(device)

    labels = labels.to(device)

    outputs = model(images)

    loss = criterion(outputs, labels)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

This is the standard PyTorch training loop.

---

# Why Labels Also Move?

Many beginners only move

```
Images
```

to GPU.

But

Loss Function compares

```
Prediction

↓

Label
```

If labels remain on CPU,

PyTorch throws an error.

---

# Common Error

```
Expected all tensors to be
on the same device
```

Reason

```
Tensor

↓

GPU

Model

↓

CPU
```

or

```
Input

↓

GPU

Labels

↓

CPU
```

Everything must be on

```
Same Device
```

---

# Wrong Example

```python
images = images.cuda()

outputs = model(images)
```

Model

```
CPU
```

Images

```
GPU
```

Runtime Error

---

# Correct Example

```python
model = model.to(device)

images = images.to(device)

labels = labels.to(device)
```

Now

Everything

```
GPU
```

---

# Checking Model Device

```python
next(model.parameters()).device
```

Output

```
cuda:0
```

Very useful while debugging.

---

# Device Independent Code

Always write

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)
```

Never

```python
device="cuda"
```

Because

Someone may run your code

without GPU.

---

# Performance Tip

Bad

```python
for i in range(1000):

    x = x.to(device)
```

Good

```python
x = x.to(device)

for i in range(1000):

    prediction = model(x)
```

Move tensors only once whenever possible.

---

# Real World Example

LLM Fine Tuning

```
Tokenizer

↓

Input IDs

↓

GPU

↓

Llama Model

↓

GPU

↓

Output

↓

Loss

↓

Backward
```

Everything stays on GPU.

If not,

Training becomes extremely slow.

---

# Best Practices

✅ Create one global device object.

✅ Move model only once.

✅ Move tensors before computation.

✅ Keep tensors on same device.

✅ Don't move tensors repeatedly.

✅ Use `.to(device)` instead of `.cuda()`.

---

# Common Mistakes

❌ Model on CPU

Tensor on GPU

---

❌ Label on CPU

Prediction on GPU

---

❌ Forgetting

```python
model.to(device)
```

---

❌ Writing

```python
device="cuda"
```

instead of

```python
torch.device(...)
```

---

# Summary

In this chapter we learned

- CPU and GPU memory
- Device object
- `.to()`
- `.cuda()`
- `.cpu()`
- Moving tensors
- Moving models
- Device-independent code
- Common runtime errors
- Best practices

You can now execute PyTorch programs efficiently on both CPU and GPU.

---

# 📝 Exercises

### Exercise 1

Create a tensor on CPU.

Print its device.

Move it to GPU.

Print again.

---

### Exercise 2

Create

```
1000 × 1000
```

tensor.

Move it to GPU.

---

### Exercise 3

Create two GPU tensors.

Add them.

Print output.

---

### Exercise 4

Move a simple neural network to GPU.

---

### Exercise 5

Print

```
next(model.parameters()).device
```

---

# 🎤 Interview Questions

1. Why do tensors need to be moved to GPU?
2. Difference between `.cuda()` and `.to()`?
3. Why is `torch.device()` recommended?
4. What causes **Expected all tensors to be on the same device**?
5. How do you move a complete model to GPU?
6. How do you move tensors back to CPU?
7. Why shouldn't `.to(device)` be called repeatedly inside a loop?
8. What is device-independent programming?

---

# Module 3 – Part 4

# Mixed Precision Training (FP32, FP16, BF16)

# 📚 Table of Contents

- Introduction
- What is Precision?
- Why Precision Matters
- Floating Point Numbers
- FP32
- FP16
- BF16
- Integer Precision
- Memory Comparison
- Speed Comparison
- Tensor Cores
- Mixed Precision
- Automatic Mixed Precision (AMP)
- Advantages
- Limitations
- Summary
- Interview Questions
- Exercises

---

# Story

Imagine you are training

```
Llama 3

7 Billion Parameters
```

Every parameter is stored inside memory.

Suppose each parameter takes

```
4 Bytes
```

Then

```
7 Billion × 4

≈ 28 GB
```

Only the weights require around **28 GB**.

Now remember,

during training we also store

- Gradients
- Activations
- Optimizer States

Memory usage easily becomes

```
60–80 GB
```

Most consumer GPUs have

```
8 GB

12 GB

16 GB
```

How can we train such large models?

The answer is

> **Mixed Precision Training**

---

# What is Precision?

Precision means

> **How accurately a number is stored inside memory.**

Example

```
3
```

Simple Integer.

Now

```
3.1415926535
```

This decimal number needs more memory.

Higher precision

↓

More memory

↓

More computation

---

# Floating Point Numbers

Computers store decimal numbers using

```
Floating Point Representation
```

General Format

```
Sign

Exponent

Mantissa
```

```
+------------+-----------+------------+
| Sign Bit   | Exponent  | Mantissa   |
+------------+-----------+------------+
```

These three parts determine

- Positive/Negative
- Scale
- Accuracy

---

# Types of Precision

Deep Learning mainly uses

```
FP64

↓

FP32

↓

FP16

↓

BF16
```

Sometimes

```
INT8

INT4
```

for inference.

---

# FP64 (Double Precision)

Memory

```
64 Bits

=

8 Bytes
```

Advantages

- Very High Precision
- Scientific Computing

Disadvantages

- Slow
- High Memory Usage

Rarely used in Deep Learning.

---

# FP32 (Single Precision)

Memory

```
32 Bits

=

4 Bytes
```

Structure

```
1 Bit

Sign

8 Bits

Exponent

23 Bits

Mantissa
```

Advantages

- High Accuracy
- Stable Training
- Standard Format

Disadvantages

- Uses More Memory

Most traditional Deep Learning models were trained using FP32.

---

# FP16 (Half Precision)

Memory

```
16 Bits

=

2 Bytes
```

Structure

```
1 Bit

Sign

5 Bits

Exponent

10 Bits

Mantissa
```

Advantages

✅ Half Memory

✅ Faster GPU Computation

✅ More Data Fits in VRAM

Disadvantages

❌ Smaller Number Range

❌ Numerical Overflow

❌ Numerical Underflow

---

# BF16 (Brain Floating Point)

Developed by

```
Google
```

Memory

```
16 Bits
```

Structure

```
1 Bit

Sign

8 Bits

Exponent

7 Bits

Mantissa
```

Notice

Exponent is same as FP32.

This means

BF16 can represent

very small

and

very large

numbers much better than FP16.

---

# FP16 vs BF16

| Feature | FP16 | BF16 |
|----------|------|-------|
| Bits | 16 |16|
| Memory |2 Bytes|2 Bytes|
| Speed |Fast|Fast|
| Numerical Stability|Medium|High|
| Used in LLMs|Yes|Yes|
| Overflow Risk|Higher|Lower|

Modern LLM training prefers

```
BF16
```

whenever supported.

---

# Memory Comparison

Suppose

```
1 Billion Parameters
```

### FP32

```
1B × 4

=

4 GB
```

---

### FP16

```
1B × 2

=

2 GB
```

---

### BF16

```
1B × 2

=

2 GB
```

Immediately

Memory becomes

```
50%

Less
```

---

# Speed Comparison

| Precision | Speed |
|-----------|--------|
| FP64 | Slowest |
| FP32 | Standard |
| FP16 | Faster |
| BF16 | Faster |

Modern GPUs

```
RTX

A100

H100

L40

H200
```

contain hardware specially optimized for FP16/BF16.

---

# Why FP16 is Faster

Smaller numbers

↓

Less Memory Transfer

↓

Higher Throughput

↓

More Parallel Operations

↓

Faster Training

Memory bandwidth is often the bottleneck in deep learning, so reducing the amount of data transferred speeds up computation.

---

# Tensor Cores

Modern NVIDIA GPUs contain

```
Tensor Cores
```

These are specialized hardware units designed specifically for matrix multiplication.

Normal CUDA Core

```
General Computation
```

Tensor Core

```
Matrix Multiplication

Deep Learning
```

Tensor Cores achieve their best performance using

- FP16
- BF16
- TF32

---

# What is Mixed Precision?

Instead of training entirely in FP32,

we combine

```
FP16

+

FP32
```

or

```
BF16

+

FP32
```

Hence the name

```
Mixed Precision
```

---

# Basic Idea

Some operations require

```
High Accuracy
```

Others only require

```
High Speed
```

So

```
Forward Pass

↓

FP16

↓

Backward Pass

↓

FP16

↓

Weight Update

↓

FP32
```

Master weights are typically kept in FP32 for numerical stability while many computations run in FP16 or BF16.

---

# Automatic Mixed Precision (AMP)

PyTorch provides

```python
torch.autocast()
```

which automatically chooses the appropriate precision for many operations.

Example

```python
with torch.autocast(device_type="cuda"):

    output = model(images)

    loss = criterion(output, labels)
```

You don't need to manually convert every tensor.

---

# Gradient Scaling

Small FP16 values can become

```
0
```

This problem is called

```
Gradient Underflow
```

PyTorch solves this using

```python
GradScaler
```

which scales gradients during training and rescales them before updating the weights.

---

# Advantages of Mixed Precision

✅ Faster Training

✅ Less GPU Memory

✅ Larger Batch Sizes

✅ Lower VRAM Usage

✅ Better GPU Utilization

✅ Essential for Large Language Models

---

# Limitations

❌ Some operations still require FP32

❌ Older GPUs may not benefit significantly

❌ Numerical instability can occur without proper scaling

---

# Real World Usage

Mixed Precision is widely used in:

- Llama
- Gemma
- Qwen
- DeepSeek
- Mistral
- Stable Diffusion
- Vision Transformers (ViT)
- CNNs such as ResNet and EfficientNet

It is considered a standard optimization in modern deep learning.

---

# Precision Summary

| Precision | Bits | Bytes | Speed | Memory | Typical Use |
|------------|------|-------|-------|--------|-------------|
| FP64 |64|8|Slow|High|Scientific Computing|
| FP32 |32|4|Normal|Medium|Standard Training|
| FP16 |16|2|Fast|Low|Mixed Precision Training|
| BF16 |16|2|Fast|Low|Modern LLM Training|
| INT8 |8|1|Very Fast|Very Low|Inference|
| INT4 |4|0.5|Extremely Fast|Lowest|Quantized LLMs|

---

# Key Takeaways

- Precision determines how numbers are represented in memory.
- Lower precision reduces memory consumption and increases throughput.
- FP32 offers high numerical accuracy and remains the baseline.
- FP16 halves memory usage but has a smaller representable range.
- BF16 keeps the wide exponent range of FP32 while using only 16 bits.
- Tensor Cores are optimized for FP16 and BF16 operations.
- Mixed Precision combines speed and stability by using multiple precisions during training.
- Automatic Mixed Precision (AMP) in PyTorch simplifies this process.

---

# 🎤 Interview Questions

1. What is floating point precision?
2. Why is FP16 faster than FP32?
3. Difference between FP16 and BF16?
4. Why is BF16 preferred for many LLMs?
5. What is Mixed Precision Training?
6. What are Tensor Cores?
7. Why do we still keep some values in FP32?
8. What is Gradient Underflow?
9. What is `torch.autocast()`?
10. Why is Mixed Precision important for training billion-parameter models?

---

# 📝 Exercises

### Exercise 1

Calculate the memory required for a model with **500 million parameters** using:

- FP32
- FP16
- BF16

---

### Exercise 2

Research whether your GPU supports

- FP16
- BF16
- Tensor Cores

---

### Exercise 3

Why would FP64 generally be a poor choice for training a large language model?



# Module 3 – Part 5

# Device Management & Best Practices


# 📚 Table of Contents

- Introduction
- What is Device Management?
- Why Device Management Matters
- CPU vs GPU Memory
- Device Objects
- Checking Available Devices
- Device Independent Programming
- Moving Tensors
- Moving Models
- Device Synchronization
- Multiple GPUs
- Device Mismatch Errors
- Memory Management
- CUDA Cache
- Performance Tips
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you trained a CNN on your laptop.

Everything works perfectly.

Now you upload the same project to

```
AWS

↓

RunPod

↓

Google Colab

↓

Kaggle
```

Suddenly your program crashes.

Error

```
Expected all tensors to be on the same device.
```

or

```
CUDA not available.
```

or

```
CUDA Out Of Memory.
```

The problem isn't your model.

The problem is **Device Management**.

A good PyTorch developer never writes code only for one machine.

They write code that runs on

- CPU
- Single GPU
- Multiple GPUs
- Cloud Servers

without changing the source code.

---

# What is Device Management?

Device Management means

> Managing where tensors and models are stored and where computations are executed.

Possible devices

```
CPU

GPU

Multiple GPUs
```

PyTorch gives complete control over device placement.

---

# Device Hierarchy

```
Computer

│

├── CPU

│

└── GPU

     │

     ├── cuda:0

     ├── cuda:1

     ├── cuda:2

     └── cuda:3
```

Large servers often contain

```
4

8

16

GPUs
```

---

# Creating a Device

Always create one global device object.

```python
import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)
```

Never hardcode

```python
device = "cuda"
```

because your code will fail on systems without a GPU.

---

# Checking Current Device

```python
print(device)
```

Output

```
cuda
```

or

```
cpu
```

---

# Number of GPUs

```python
torch.cuda.device_count()
```

Example

```
2
```

Meaning

```
cuda:0

cuda:1
```

---

# Current GPU

```python
torch.cuda.current_device()
```

Output

```
0
```

---

# GPU Name

```python
torch.cuda.get_device_name(0)
```

Output

```
NVIDIA RTX 4060
```

---

# Tensor Placement

CPU Tensor

```python
x = torch.tensor([1,2,3])
```

Move

```python
x = x.to(device)
```

Now

```
CPU

↓

GPU
```

---

# Model Placement

Every model has parameters.

Those parameters also live on a device.

```python
model = MyModel()

model = model.to(device)
```

Now every layer moves automatically.

---

# Data Loader

Inside training

Always move

```
Images

Labels
```

Example

```python
for images, labels in train_loader:

    images = images.to(device)

    labels = labels.to(device)
```

---

# Complete Workflow

```
Dataset

↓

DataLoader

↓

Tensor

↓

GPU

↓

Model

↓

Prediction

↓

Loss

↓

Backward

↓

Optimizer
```

---

# Device Independent Programming

Good Code

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)
```

Bad Code

```python
device="cuda"
```

Reason

Some users don't have GPUs.

---

# Why is .to(device) Preferred?

Instead of

```python
.cuda()
```

PyTorch recommends

```python
.to(device)
```

Advantages

- Cleaner
- Portable
- CPU Compatible
- Cloud Compatible

---

# Checking Tensor Device

```python
print(x.device)
```

Output

```
cuda:0
```

---

# Checking Model Device

```python
next(model.parameters()).device
```

Useful for debugging.

---

# Device Mismatch Error

Most Common Error

```
Expected all tensors to be on the same device.
```

Example

```
Tensor

↓

GPU

Model

↓

CPU
```

PyTorch cannot perform computation across different devices.

---

# Wrong Code

```python
images = images.cuda()

outputs = model(images)
```

Model

```
CPU
```

Images

```
GPU
```

Runtime Error.

---

# Correct Code

```python
model = model.to(device)

images = images.to(device)

labels = labels.to(device)
```

Everything lives on

```
Same Device
```

---

# CUDA Synchronization

GPU operations are asynchronous.

This means

```
Python

↓

Continues

↓

GPU Still Working
```

Sometimes we need to wait.

```python
torch.cuda.synchronize()
```

Useful while

- Benchmarking
- Measuring Time

---

# GPU Memory

Every GPU has

```
VRAM
```

Used for

- Tensors
- Model Weights
- Gradients
- Optimizer States
- Activations

---

# Memory Usage

Current Memory

```python
torch.cuda.memory_allocated()
```

Reserved Memory

```python
torch.cuda.memory_reserved()
```

Maximum Memory

```python
torch.cuda.max_memory_allocated()
```

These functions help monitor VRAM usage during training.

---

# Clearing Cache

Sometimes memory isn't immediately released.

PyTorch caches memory for faster reuse.

You can clear unused cached memory with

```python
torch.cuda.empty_cache()
```

> Note: This clears cached memory, not memory currently used by active tensors.

---

# Out of Memory (OOM)

Example

```
CUDA Out Of Memory
```

Reasons

- Batch size too large
- Model too large
- Memory leak
- Multiple programs using GPU

---

# Fix OOM

✅ Reduce Batch Size

✅ Use FP16

✅ Use BF16

✅ Gradient Accumulation

✅ Gradient Checkpointing

✅ Close other GPU applications

---

# Monitoring GPU

Terminal

```bash
nvidia-smi
```

Shows

- GPU Usage
- Temperature
- VRAM
- Running Processes
- Power Usage

Useful while training.

---

# Multiple GPUs

Example

```
cuda:0

cuda:1

cuda:2

cuda:3
```

PyTorch supports

- Data Parallel
- Distributed Data Parallel (DDP)

DDP is the recommended approach for modern large-scale training.

---

# Good Training Loop

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

model = MyModel().to(device)

for images, labels in train_loader:

    images = images.to(device)

    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()
```

This pattern is used in most PyTorch projects.

---

# Best Practices

✅ Create one global device object.

---

✅ Move model once.

---

✅ Move data inside training loop.

---

✅ Use `.to(device)`.

---

✅ Monitor GPU memory.

---

✅ Use Mixed Precision whenever possible.

---

✅ Keep tensors on same device.

---

# Common Mistakes

❌ Hardcoding

```python
device="cuda"
```

---

❌ Forgetting

```python
model.to(device)
```

---

❌ Labels on CPU.

---

❌ Calling

```python
.to(device)
```

inside every operation unnecessarily.

---

❌ Ignoring GPU memory usage.

---

# Real World Example

Fine Tuning Llama

```
Dataset

↓

Tokenizer

↓

Input IDs

↓

GPU

↓

Llama Model

↓

GPU

↓

Loss

↓

Backward

↓

Optimizer

↓

Save Checkpoint
```

Everything remains on the GPU for efficient training.

---

# Cheat Sheet

| Task | Function |
|------|----------|
| Current Device | `torch.cuda.current_device()` |
| GPU Name | `torch.cuda.get_device_name(0)` |
| GPU Count | `torch.cuda.device_count()` |
| CUDA Available | `torch.cuda.is_available()` |
| Move Tensor | `.to(device)` |
| Move Model | `model.to(device)` |
| CPU | `.cpu()` |
| GPU | `.cuda()` |
| Clear Cache | `torch.cuda.empty_cache()` |
| Sync GPU | `torch.cuda.synchronize()` |

---

# Summary

In this chapter, you learned:

- What device management is.
- How to create and use a `torch.device`.
- How to move tensors and models between CPU and GPU.
- How to write device-independent PyTorch code.
- How to monitor and manage GPU memory.
- Common CUDA errors and how to resolve them.
- Best practices for writing efficient and portable training code.

Mastering device management is essential for building reliable PyTorch applications that run seamlessly on local machines, cloud GPUs, and production servers.

---

# 🎤 Interview Questions

1. What is device management in PyTorch?
2. Why should you use `torch.device()` instead of hardcoding `"cuda"`?
3. What causes the "Expected all tensors to be on the same device" error?
4. How do you move a model to the GPU?
5. What is the difference between `.cuda()` and `.to(device)`?
6. What does `torch.cuda.empty_cache()` do?
7. How can you check current GPU memory usage?
8. Why are GPU operations asynchronous?
9. How do you monitor GPU usage while training?
10. What is the recommended approach for multi-GPU training?

---

# 📝 Exercises

### Exercise 1

Create a tensor on the CPU and move it to the GPU.

---

### Exercise 2

Print your GPU name and CUDA version.

---

### Exercise 3

Create a neural network and move it to the selected device.

---

### Exercise 4

Print the device of the first model parameter.

---

### Exercise 5

Run `nvidia-smi` while training a model and observe:

- GPU utilization
- Memory usage
- Temperature

---
