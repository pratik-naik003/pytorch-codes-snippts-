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

# 📘 PyTorch Complete Course

# Module 4 – Automatic Differentiation (Autograd)

> **Level:** Beginner → Intermediate

> **Prerequisites**
>
> - PyTorch Installation
> - Tensors
> - CUDA Basics

---

# 📚 Table of Contents

- Introduction
- Story
- What is Autograd?
- Why Autograd?
- Computational Graph
- Forward Pass
- Backward Pass
- Chain Rule
- requires_grad
- grad
- backward()
- Gradient Accumulation
- zero_grad()
- detach()
- torch.no_grad()
- Optimizer Flow
- Custom Autograd
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# Story

Imagine you're training a neural network.

```
Image

↓

Prediction

↓

Loss

↓

Update Weights
```

But...

How does PyTorch know

```
Which weight

Should change

And

By how much?
```

Do we manually calculate

```
∂Loss/∂W1

∂Loss/∂W2

∂Loss/∂W3

...

Millions of derivatives?
```

Imagine Llama-3 with

```
8 Billion Parameters
```

Would you calculate

```
8 Billion derivatives
```

manually?

Impossible.

This is exactly why **Autograd** exists.

---

# What is Autograd?

Autograd is PyTorch's **automatic differentiation engine**.

It automatically

- Tracks operations
- Builds a computation graph
- Computes gradients
- Applies the Chain Rule

without us writing any calculus.

---

# What is a Gradient?

A gradient tells us

> **How much a parameter should change to reduce the loss.**

Imagine climbing a mountain.

```
Current Position

↓

Slope

↓

Move Downhill
```

Gradient is simply the slope.

Large Gradient

↓

Large Update

Small Gradient

↓

Small Update

---

# Why Do We Need Gradients?

Suppose

```
Prediction = 10

Actual = 15
```

Loss

```
5
```

Should we

Increase weight?

Decrease weight?

Keep it same?

Gradient tells us the answer.

---

# Training Flow

```
Input

↓

Prediction

↓

Loss

↓

Gradient

↓

Weight Update

↓

Better Prediction
```

This repeats thousands of times.

---

# What is a Computational Graph?

Whenever we perform tensor operations,

PyTorch silently creates a graph.

Example

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3

z = y + 4
```

Internally

```
x

↓

×

3

↓

+

4

↓

z
```

PyTorch remembers

every operation.

---

# Why Build This Graph?

Later,

when we call

```python
z.backward()
```

PyTorch walks backwards

through this graph

computing gradients.

Hence the name

```
Backpropagation
```

---

# Forward Pass

Forward Pass means

```
Input

↓

Neural Network

↓

Prediction
```

Example

```python
x = torch.tensor(2.0)

y = x * 5

print(y)
```

Output

```
10
```

Simple.

---

# Backward Pass

Backward Pass computes gradients.

Example

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

Output

```
tensor(4.)
```

Why?

Because

```
y = x²

dy/dx = 2x

2 × 2 = 4
```

PyTorch did the calculus automatically.

---

# Chain Rule

Suppose

```
x

↓

×

3

↓

+

5

↓

Square
```

Instead of computing everything together,

PyTorch applies

```
Chain Rule
```

step by step.

This allows neural networks with millions of operations to compute gradients efficiently.

---

# requires_grad

Most important parameter.

Default

```python
False
```

Example

```python
a = torch.tensor(5.)

print(a.requires_grad)
```

Output

```
False
```

Now

```python
a = torch.tensor(
    5.,
    requires_grad=True
)
```

Output

```
True
```

Now PyTorch tracks every operation.

---

# Example

```python
x = torch.tensor(
    2.,
    requires_grad=True
)

y = x * 4

print(y)
```

Output

```
tensor(8.,
grad_fn=<MulBackward0>)
```

Notice

```
grad_fn
```

This means

Autograd is tracking the operation.

---

# grad_fn

Every operation stores

```
History
```

Example

```
MulBackward

AddBackward

PowBackward

ReluBackward
```

These functions help build the computation graph.

---

# Leaf Tensor

A tensor directly created by the user with `requires_grad=True` is called a **leaf tensor**.

Example

```python
x = torch.tensor(3.0, requires_grad=True)
```

`x` is a leaf tensor.

---

# Non-Leaf Tensor

```python
y = x * 2
```

`y` is not a leaf tensor.

It was created from another tensor.

---

# Computing Gradient

Example

```python
x = torch.tensor(
    3.,
    requires_grad=True
)

y = x ** 2

y.backward()

print(x.grad)
```

Output

```
6
```

Because

```
dy/dx = 2x

2 × 3 = 6
```

---

# Multiple Operations

```python
x = torch.tensor(
    2.,
    requires_grad=True
)

y = x * 5

z = y ** 2

z.backward()

print(x.grad)
```

Calculation

```
z=(5x)^2

=25x²

dz/dx

=50x

x=2

100
```

Output

```
100
```

---

# Gradient Accumulation

One very important concept.

Example

```python
x = torch.tensor(
    2.,
    requires_grad=True
)

y = x ** 2

y.backward()

print(x.grad)
```

Output

```
4
```

Now

```python
y = x ** 2

y.backward()

print(x.grad)
```

Output

```
8
```

Why?

Because

PyTorch **adds** gradients.

It does NOT replace them.

---

# Why Accumulate?

Useful for

Large Models

Small GPUs

Gradient Accumulation simulates larger batch sizes by summing gradients over multiple forward/backward passes before updating the weights.

---

# zero_grad()

To reset gradients

```python
optimizer.zero_grad()
```

Without this

Gradients continue accumulating.

---

# Training Loop

```python
for epoch in range(epochs):

    optimizer.zero_grad()

    output=model(x)

    loss=criterion(output,y)

    loss.backward()

    optimizer.step()
```

This is the heart of every PyTorch model.

---

# detach()

Sometimes

we don't want gradients.

Example

```python
x=torch.tensor(
    5.,
    requires_grad=True
)

y=x.detach()

print(y.requires_grad)
```

Output

```
False
```

Detached tensors are removed from the computation graph.

---

# Why detach()?

Useful when

- Logging values
- Saving predictions
- Converting to NumPy
- Freezing part of a computation

---

# torch.no_grad()

Turns off gradient tracking.

Example

```python
with torch.no_grad():

    prediction=model(images)
```

Everything inside the block is executed without building a computation graph.

---

# Why Use no_grad()?

During

```
Testing

Inference

Validation
```

We don't train.

So gradients are unnecessary.

Benefits

- Faster
- Less Memory
- Better Performance

---

# Difference

| detach() | no_grad() |
|----------|-----------|
| Single Tensor | Entire Block |
| Removes Graph | Doesn't Build Graph |
| Used During Computation | Used During Inference |

---

# Optimizer Flow

Complete Training Pipeline

```
Input

↓

Forward Pass

↓

Prediction

↓

Loss

↓

Backward()

↓

Gradients

↓

Optimizer.step()

↓

Weights Updated

↓

Repeat
```

---

# optimizer.step()

Updates weights.

Example

```python
optimizer.step()
```

Internally

```
New Weight

=

Old Weight

-

Learning Rate

×

Gradient
```

---

# Custom Autograd

PyTorch even allows you to create your own differentiable operations.

```python
from torch.autograd import Function

class Square(Function):

    @staticmethod
    def forward(ctx, x):

        ctx.save_for_backward(x)

        return x*x

    @staticmethod
    def backward(ctx, grad_output):

        x, = ctx.saved_tensors

        return grad_output * 2 * x
```

This is useful for advanced research and custom mathematical operations.

---

# Common Mistakes

❌ Forgetting `requires_grad=True`

❌ Forgetting `optimizer.zero_grad()`

❌ Calling `backward()` twice without `retain_graph=True`

❌ Using `.numpy()` on a tensor that requires gradients (detach it first)

❌ Performing inference without `torch.no_grad()`

---

# Best Practices

✅ Use `requires_grad=True` only for trainable parameters.

✅ Always call `optimizer.zero_grad()` before `backward()`.

✅ Use `torch.no_grad()` during inference.

✅ Use `detach()` when gradients are not needed.

✅ Inspect `.grad` while debugging.

---

# Summary

- Autograd automatically computes gradients.
- Gradients tell the optimizer how to update weights.
- Computational graphs store every operation.
- `requires_grad=True` enables tracking.
- `backward()` computes gradients.
- Gradients accumulate unless cleared.
- `detach()` removes tensors from the graph.
- `torch.no_grad()` disables gradient tracking for inference.
- `optimizer.step()` updates model parameters.

---

# 🎤 Interview Questions

1. What is Autograd?
2. What is a computational graph?
3. Why do we need gradients?
4. What does `requires_grad=True` do?
5. What is `grad_fn`?
6. Difference between leaf and non-leaf tensors?
7. Why do gradients accumulate?
8. What does `optimizer.zero_grad()` do?
9. Difference between `detach()` and `torch.no_grad()`?
10. Explain the complete training flow in PyTorch.

---

# 📝 Exercises

1. Create a tensor with `requires_grad=True` and compute its square.
2. Verify gradient accumulation by calling `backward()` twice.
3. Reset gradients using `optimizer.zero_grad()`.
4. Compare `detach()` and `torch.no_grad()` with small examples.
5. Build a tiny linear model and inspect gradients after one training step.

---

# Module 5 – Building Neural Networks with `torch.nn`

> **Goal:** Learn how to build neural networks in PyTorch using the `torch.nn` module.

---

# 📚 Table of Contents

- Introduction
- What is `torch.nn`?
- Why Use `torch.nn`?
- `nn.Module`
- Building Your First Neural Network
- Layers
- Activation Functions
- Forward Pass
- Model Parameters
- Model Summary
- `nn.Sequential`
- Custom Neural Networks
- Training Workflow
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Introduction

So far, you've learned:

- Tensors
- CUDA
- Autograd

Now it's time to build **actual neural networks**.

Without `torch.nn`, you would have to manually create:

- Weights
- Biases
- Forward functions
- Gradient calculations

PyTorch simplifies all of this with the **`torch.nn`** module.

---

# What is `torch.nn`?

`torch.nn` is a module that provides everything required to build neural networks.

It includes:

- Layers
- Activation Functions
- Loss Functions
- Containers
- Model Utilities

Instead of writing neural networks from scratch, we use ready-made building blocks.

---

# Why Use `torch.nn`?

Without `torch.nn`

```text
Create Weights

↓

Create Biases

↓

Write Forward Pass

↓

Compute Gradients

↓

Update Parameters
```

With `torch.nn`

```text
Define Layers

↓

Write Forward()

↓

Train Model
```

PyTorch automatically manages parameters and integrates with Autograd.

---

# What is `nn.Module`?

Every neural network in PyTorch inherits from `nn.Module`.

Think of `nn.Module` as the **base class** for all models.

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```

Every custom model starts like this.

---

# Structure of an `nn.Module`

Every model has two main methods:

## 1. `__init__()`

Used to define:

- Layers
- Trainable Parameters

## 2. `forward()`

Defines how data flows through the model.

```text
Input

↓

Layer 1

↓

Activation

↓

Layer 2

↓

Output
```

---

# Your First Neural Network

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

---

# Creating the Model

```python
model = SimpleNN()

print(model)
```

Output

```text
SimpleNN(
  (fc1): Linear(in_features=4, out_features=8)
  (fc2): Linear(in_features=8, out_features=2)
)
```

---

# What is a Layer?

A layer transforms input data into another representation.

Example:

```text
Input

↓

Linear Layer

↓

Activation

↓

Output
```

Common layers:

- `nn.Linear`
- `nn.Conv2d`
- `nn.Embedding`
- `nn.LSTM`
- `nn.BatchNorm`
- `nn.Dropout`

---

# Linear Layer

The most basic layer.

Mathematical equation:

\[
y = Wx + b
\]

PyTorch:

```python
layer = nn.Linear(5, 3)
```

Meaning:

- 5 input features
- 3 output features

---

# Passing Data Through a Layer

```python
import torch

x = torch.randn(2, 5)

layer = nn.Linear(5, 3)

output = layer(x)

print(output.shape)
```

Output:

```text
torch.Size([2, 3])
```

---

# Activation Functions

A neural network without activation functions behaves like a single linear transformation.

Activation functions introduce **non-linearity**.

Common activations:

| Function | Use Case |
|----------|----------|
| ReLU | Most common |
| Sigmoid | Binary classification |
| Tanh | Older networks |
| GELU | Transformers |
| Softmax | Multi-class output |

---

# ReLU

```python
relu = nn.ReLU()

x = torch.tensor([-2.0, 0.0, 3.0])

print(relu(x))
```

Output

```text
tensor([0., 0., 3.])
```

Formula:

\[
ReLU(x)=\max(0,x)
\]

---

# Sigmoid

Maps values between **0 and 1**.

```python
sigmoid = nn.Sigmoid()
```

Commonly used for binary classification.

---

# Softmax

Converts outputs into probabilities.

```python
softmax = nn.Softmax(dim=1)
```

Example:

```text
Before

[2.1, 1.3, 0.5]

↓

After

[0.62, 0.27, 0.11]
```

---

# GELU

Used in modern Transformer models like:

- BERT
- GPT
- Llama
- Gemma
- Qwen

```python
gelu = nn.GELU()
```

---

# Forward Pass

The `forward()` function defines how input flows through the network.

```python
def forward(self, x):

    x = self.fc1(x)

    x = torch.relu(x)

    x = self.fc2(x)

    return x
```

Never call `forward()` directly.

Instead:

```python
output = model(x)
```

PyTorch automatically calls `forward()`.

---

# Model Parameters

Every layer has trainable parameters.

Print them:

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

Example output:

```text
fc1.weight torch.Size([8,4])
fc1.bias torch.Size([8])
fc2.weight torch.Size([2,8])
fc2.bias torch.Size([2])
```

---

# Counting Parameters

```python
total = sum(p.numel() for p in model.parameters())

print(total)
```

Useful for comparing model sizes.

---

# Using `nn.Sequential`

For simple networks:

```python
model = nn.Sequential(

    nn.Linear(4,8),

    nn.ReLU(),

    nn.Linear(8,2)
)
```

Advantages:

- Short
- Clean
- Easy to read

Limitations:

- Less flexible than custom classes.

---

# Custom Neural Networks

Preferred for real-world projects.

```python
class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Linear(10, 20)

        self.output = nn.Linear(20, 3)

    def forward(self, x):

        x = torch.relu(self.features(x))

        return self.output(x)
```

---

# Typical Training Workflow

```text
Input Data

↓

Model

↓

Prediction

↓

Loss Function

↓

loss.backward()

↓

Optimizer.step()

↓

Updated Weights
```

---

# Example Workflow

```python
model = SimpleNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Training:

```python
output = model(x)

loss = criterion(output, y)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

---

# Model Modes

Training mode:

```python
model.train()
```

Evaluation mode:

```python
model.eval()
```

Why?

Some layers behave differently during training and inference.

Examples:

- Dropout
- BatchNorm

---

# Saving Model

```python
torch.save(
    model.state_dict(),
    "model.pth"
)
```

Loading:

```python
model.load_state_dict(
    torch.load("model.pth")
)
```

---

# Best Practices

✅ Inherit from `nn.Module`

✅ Keep layers inside `__init__()`

✅ Define computations only inside `forward()`

✅ Use `model(x)` instead of `model.forward(x)`

✅ Call `model.train()` during training

✅ Call `model.eval()` during inference

---

# Common Mistakes

❌ Forgetting `super().__init__()`

❌ Creating layers inside `forward()`

❌ Calling `forward()` directly

❌ Forgetting `model.eval()` during inference

❌ Forgetting `optimizer.zero_grad()`

---

# Summary

- `torch.nn` provides building blocks for neural networks.
- Every model inherits from `nn.Module`.
- Layers are defined in `__init__()`.
- The computation is defined in `forward()`.
- `nn.Linear` performs linear transformations.
- Activation functions introduce non-linearity.
- `nn.Sequential` is useful for simple models.
- Custom classes are preferred for complex architectures.
- Parameters are updated using the optimizer after backpropagation.

---

# 🎤 Interview Questions

1. What is `nn.Module`?
2. Why do we inherit from `nn.Module`?
3. What is the difference between `__init__()` and `forward()`?
4. Why should we use `model(x)` instead of `model.forward(x)`?
5. What does `nn.Linear` do?
6. Why are activation functions important?
7. When should you use `nn.Sequential`?
8. What is the purpose of `model.train()` and `model.eval()`?
9. How do you count the number of trainable parameters?
10. How do you save and load a PyTorch model?

---

# 📝 Exercises

1. Build a neural network with:
   - Input: 10 features
   - Hidden Layer: 32 neurons
   - Output: 5 classes

2. Print all model parameters and their shapes.

3. Count the total number of trainable parameters.

4. Replace `ReLU` with `GELU` and observe the code changes.

5. Build the same network using `nn.Sequential`.

-
# Module 6 – Loss Functions (`torch.nn`)

> **Goal:** Learn what loss functions are, why they are important, how they work mathematically, and how to use them correctly in PyTorch.

---

# 📚 Table of Contents

- Introduction
- What is a Loss Function?
- Why Do We Need Loss Functions?
- Loss vs Cost
- How Training Works
- Regression Losses
- Classification Losses
- Binary Classification Losses
- Cross Entropy Loss
- BCE Loss
- BCEWithLogitsLoss
- MSE Loss
- L1 Loss
- Huber Loss
- KL Divergence Loss
- Choosing the Right Loss Function
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you're teaching a child to solve math problems.

The child answers

```
5 + 5 = 12
```

You immediately know

```
Wrong Answer
```

But the child asks

> "How wrong am I?"

Simply saying

```
Wrong
```

isn't enough.

You need to tell

- How far from the correct answer
- Whether the answer is improving
- Whether learning is happening

Deep Learning works exactly the same way.

The neural network makes a prediction.

Now we need a mathematical function that tells

```
How bad

or

How good

the prediction is.
```

That function is called the

> **Loss Function**

---

# What is Loss?

Loss measures

> **The difference between the predicted output and the actual output.**

Smaller Loss

↓

Better Prediction

Larger Loss

↓

Poor Prediction

Goal of every neural network

```
Minimize Loss
```

---

# Training Pipeline

```
Input

↓

Model

↓

Prediction

↓

Loss Function

↓

Loss

↓

Backward()

↓

Optimizer

↓

Update Weights
```

Without a loss function

the neural network

cannot learn.

---

# Example

Suppose

Actual House Price

```
₹50,00,000
```

Prediction

```
₹49,80,000
```

Loss

Small

Now

Prediction

```
₹10,00,000
```

Loss

Very Large

The optimizer will try to reduce this error.

---

# Loss vs Cost

People often use these terms interchangeably.

Technically

Loss

```
One Training Example
```

Cost

```
Average Loss

Entire Dataset
```

Most libraries simply call everything

```
Loss
```

---

# Types of Problems

Machine Learning problems are mainly divided into

```
Regression

Classification
```

Different problems require different loss functions.

---

# Regression Losses

Used when predicting

continuous values.

Examples

- House Price
- Temperature
- Salary
- Stock Price

Popular Losses

- MSELoss
- L1Loss
- SmoothL1Loss

---

# Classification Losses

Used when predicting

categories.

Examples

- Dog vs Cat
- Spam Detection
- Digit Recognition
- Disease Detection

Popular Losses

- CrossEntropyLoss
- BCELoss
- BCEWithLogitsLoss
- NLLLoss

---

# Mean Squared Error (MSELoss)

Most common regression loss.

Formula

\[
MSE=\frac1n\sum(y-\hat y)^2
\]

PyTorch

```python
import torch
import torch.nn as nn

criterion = nn.MSELoss()

prediction = torch.tensor([2.5])

target = torch.tensor([3.0])

loss = criterion(prediction, target)

print(loss)
```

Advantages

✅ Easy

✅ Smooth

Disadvantages

❌ Sensitive to outliers

---

# L1 Loss (Mean Absolute Error)

Formula

\[
|y-\hat y|
\]

PyTorch

```python
criterion = nn.L1Loss()
```

Advantages

- Robust to outliers

Disadvantages

- Less smooth gradients

---

# SmoothL1Loss (Huber Loss)

Combination of

```
MSE

+

L1
```

PyTorch

```python
criterion = nn.SmoothL1Loss()
```

Used in

- Object Detection
- Bounding Box Regression

---

# Binary Classification

Example

```
Spam

Not Spam
```

Only two classes.

Popular losses

- BCELoss
- BCEWithLogitsLoss

---

# BCELoss

Used after

```
Sigmoid
```

Example

```python
model

↓

Sigmoid

↓

BCELoss
```

PyTorch

```python
criterion = nn.BCELoss()
```

---

# BCEWithLogitsLoss

Recommended instead of BCELoss.

Why?

It combines

```
Sigmoid

+

BCELoss
```

into one stable operation.

PyTorch

```python
criterion = nn.BCEWithLogitsLoss()
```

Advantages

- Numerically Stable
- Faster
- Less Overflow

---

# Multi-Class Classification

Example

```
Cat

Dog

Horse

Lion
```

Only one correct answer.

Use

```
CrossEntropyLoss
```

---

# CrossEntropyLoss

Most popular classification loss.

Used in

- CNN
- ResNet
- Vision Transformer
- Llama Classifiers
- BERT Classification

PyTorch

```python
criterion = nn.CrossEntropyLoss()
```

Example

```python
output = model(images)

loss = criterion(output, labels)
```

Important

Do **NOT** apply Softmax before `CrossEntropyLoss`.

PyTorch applies it internally.

---

# Why No Softmax?

Wrong

```python
output = torch.softmax(output)

loss = criterion(output, labels)
```

Correct

```python
loss = criterion(output, labels)
```

---

# NLLLoss

Negative Log Likelihood Loss.

Requires

```
LogSoftmax

↓

NLLLoss
```

PyTorch

```python
criterion = nn.NLLLoss()
```

---

# KL Divergence Loss

Measures

Difference between

two probability distributions.

Used in

- Knowledge Distillation
- Variational Autoencoders
- LLM Training
- Reinforcement Learning

PyTorch

```python
criterion = nn.KLDivLoss()
```

---

# Choosing the Right Loss

| Problem | Loss |
|----------|------|
| Regression | MSELoss |
| Regression with Outliers | L1Loss |
| Object Detection | SmoothL1Loss |
| Binary Classification | BCEWithLogitsLoss |
| Multi-Class Classification | CrossEntropyLoss |
| Probability Distribution | KLDivLoss |

---

# Complete Example

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

prediction = torch.tensor([
    [2.1,1.2,0.5]
])

target = torch.tensor([0])

loss = criterion(prediction,target)

print(loss)
```

---

# Training Loop

```python
for images, labels in train_loader:

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()
```

Loss is calculated

every iteration.

---

# Visual Workflow

```
Images

↓

Model

↓

Prediction

↓

Loss Function

↓

Loss Value

↓

Backward

↓

Optimizer

↓

Updated Model
```

---

# Best Practices

✅ Use CrossEntropyLoss for multi-class classification.

✅ Use BCEWithLogitsLoss for binary classification.

✅ Use MSELoss for regression.

✅ Don't apply Softmax before CrossEntropyLoss.

✅ Match target shapes with predictions.

---

# Common Mistakes

❌ Using MSELoss for classification.

❌ Applying Softmax before CrossEntropyLoss.

❌ Using BCELoss without Sigmoid.

❌ Wrong target datatype.

---

# Summary

- Loss functions measure prediction error.
- Smaller loss indicates better predictions.
- Regression and classification use different loss functions.
- MSELoss is common for regression.
- CrossEntropyLoss is standard for multi-class classification.
- BCEWithLogitsLoss is preferred for binary classification.
- Choosing the correct loss function is critical for successful training.

---

# 🎤 Interview Questions

1. What is a loss function?
2. Difference between loss and cost?
3. Why do we need loss functions?
4. Difference between MSELoss and L1Loss?
5. Why is CrossEntropyLoss so popular?
6. Why shouldn't we apply Softmax before CrossEntropyLoss?
7. Difference between BCELoss and BCEWithLogitsLoss?
8. What is KL Divergence?
9. Which loss is used for regression?
10. Which loss is used for binary classification?

---

# 📝 Exercises

### Exercise 1

Implement MSELoss on two tensors.

---

### Exercise 2

Use CrossEntropyLoss on a dummy classifier.

---

### Exercise 3

Compare BCELoss and BCEWithLogitsLoss.

---

### Exercise 4

Train a simple Linear Regression model using MSELoss.

---

### Exercise 5

Train a binary classifier using BCEWithLogitsLoss.

---

# Module 7 – Optimizers (`torch.optim`)

> **Goal:** Learn how neural networks update their weights using optimization algorithms and understand the most commonly used optimizers in PyTorch.

---

# 📚 Table of Contents

- Introduction
- What is an Optimizer?
- Why Do We Need Optimizers?
- Gradient Descent
- Types of Gradient Descent
- Learning Rate
- SGD
- Momentum
- RMSProp
- Adam
- AdamW
- Weight Decay
- Choosing an Optimizer
- Complete Training Loop
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you're standing on the top of a mountain.

Your goal is to reach the lowest point.

But there's one problem...

You are blindfolded.

You cannot see the valley.

The only information you have is

```
Which direction is downhill?
```

Every small step you take brings you closer to the bottom.

This is exactly how neural networks learn.

Instead of climbing a mountain,

they are trying to minimize a **Loss Function**.

The optimizer acts as the guide.

It tells the model

> **"Move your weights in this direction to reduce the loss."**

---

# What is an Optimizer?

An **Optimizer** is an algorithm that updates the model's parameters (weights and biases) using the gradients calculated during backpropagation.

Simple Workflow

```text
Input

↓

Prediction

↓

Loss

↓

Backward()

↓

Gradients

↓

Optimizer

↓

Updated Weights
```

Without an optimizer, the model cannot improve.

---

# Why Do We Need an Optimizer?

Suppose your model predicts

```
Prediction = 120

Actual = 100
```

Loss

```
20
```

Autograd computes the gradient.

But...

Who changes the weights?

The answer is

```
Optimizer
```

---

# Gradient Descent

The most fundamental optimization algorithm.

Goal

```
Reduce Loss
```

Formula

\[
W_{new}=W_{old}-LearningRate \times Gradient
\]

Meaning

```
New Weight

=

Old Weight

-

Small Step
```

---

# Visual Representation

```
Loss

↑

|

|        •

|      •

|    •

|  •

|•

+--------------------→ Weight
```

Each optimizer step moves closer to the minimum.

---

# Types of Gradient Descent

### 1. Batch Gradient Descent

Uses the **entire dataset** before updating weights.

Advantages

- Stable

Disadvantages

- Slow

---

### 2. Stochastic Gradient Descent (SGD)

Uses **one training sample** at a time.

Advantages

- Fast

Disadvantages

- Noisy updates

---

### 3. Mini-Batch Gradient Descent

Uses a **small batch** of samples.

Example

```
Batch Size = 32
```

Most Deep Learning models use Mini-Batch Gradient Descent.

---

# Learning Rate

Learning Rate determines

> **How large each optimization step should be.**

Notation

```
lr
```

Example

```python
lr = 0.001
```

---

# Choosing Learning Rate

Too Small

```
Very Slow Training
```

Too Large

```
Training becomes unstable

Loss may increase

Model may never converge
```

Good Learning Rate

```
Fast

Stable
```

---

# Stochastic Gradient Descent (SGD)

Most basic optimizer.

PyTorch

```python
import torch.optim as optim

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)
```

Advantages

- Simple
- Less Memory
- Easy to Understand

Disadvantages

- Can converge slowly
- May get stuck in local minima

---

# SGD with Momentum

Momentum helps the optimizer maintain direction and reduces oscillations.

Analogy

Imagine pushing a heavy ball downhill.

The ball gains momentum and rolls more smoothly.

PyTorch

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

Advantages

- Faster convergence
- Smoother updates
- Escapes shallow local minima more easily

---

# RMSProp

RMSProp adjusts the learning rate for each parameter individually.

Idea

```
Frequently changing weights

↓

Smaller updates

Less changing weights

↓

Larger updates
```

PyTorch

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001
)
```

Used in

- RNNs
- LSTMs
- Reinforcement Learning

---

# Adam Optimizer

Adam stands for

```
Adaptive Moment Estimation
```

Adam combines ideas from

- Momentum
- RMSProp

It is the default optimizer for many deep learning projects.

PyTorch

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Advantages

✅ Fast

✅ Stable

✅ Works well for most problems

---

# AdamW Optimizer

AdamW is an improved version of Adam.

Main Difference

```
Adam

↓

Weight Decay implemented differently

↓

Better Regularization
```

PyTorch

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001
)
```

Used in

- BERT
- GPT
- Llama
- Gemma
- Qwen
- Vision Transformers

Today, AdamW is the standard optimizer for training Transformer models.

---

# Weight Decay

Weight Decay is a regularization technique.

Purpose

Prevent the model from becoming too complex.

PyTorch

```python
optimizer = optim.AdamW(

    model.parameters(),

    lr=0.001,

    weight_decay=0.01
)
```

Benefits

- Reduces Overfitting
- Better Generalization

---

# Optimizer Comparison

| Optimizer | Speed | Memory | Common Use |
|------------|--------|---------|------------|
| SGD | Medium | Low | CNNs |
| SGD + Momentum | Fast | Low | CNNs |
| RMSProp | Fast | Medium | RNNs |
| Adam | Very Fast | Medium | General DL |
| AdamW | Very Fast | Medium | Transformers & LLMs |

---

# Creating an Optimizer

```python
import torch.optim as optim

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001
)
```

---

# What is `model.parameters()`?

Every neural network contains trainable weights.

Example

```python
for param in model.parameters():

    print(param.shape)
```

The optimizer receives these parameters and updates them after every backward pass.

---

# Optimizer Workflow

```
Input

↓

Forward Pass

↓

Prediction

↓

Loss

↓

loss.backward()

↓

Gradients

↓

optimizer.step()

↓

Updated Weights
```

---

# Complete Training Example

```python
for images, labels in train_loader:

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()
```

This is the standard training loop in PyTorch.

---

# Why `optimizer.zero_grad()`?

Gradients accumulate by default.

Without resetting them

```python
optimizer.zero_grad()
```

the gradients from previous iterations would continue to add up, leading to incorrect updates.

---

# Checking Learning Rate

```python
print(
    optimizer.param_groups[0]["lr"]
)
```

Output

```
0.001
```

Useful for debugging and learning rate scheduling.

---

# When to Use Which Optimizer?

| Task | Optimizer |
|------|-----------|
| Beginner Projects | Adam |
| CNNs | SGD + Momentum |
| RNNs | RMSProp |
| Transformers | AdamW |
| LLM Fine-Tuning | AdamW |

---

# Real World Examples

### ResNet

```
Optimizer

↓

SGD + Momentum
```

---

### Llama 3

```
Optimizer

↓

AdamW
```

---

### BERT

```
Optimizer

↓

AdamW
```

---

### Stable Diffusion

```
Optimizer

↓

AdamW
```

---

# Best Practices

✅ Start with Adam for new projects.

✅ Use AdamW for Transformer-based models.

✅ Tune the learning rate before changing optimizers.

✅ Call `optimizer.zero_grad()` before `loss.backward()`.

✅ Save optimizer state when saving checkpoints.

---

# Saving Optimizer

```python
torch.save({

    "model": model.state_dict(),

    "optimizer": optimizer.state_dict()

}, "checkpoint.pth")
```

Loading

```python
checkpoint = torch.load("checkpoint.pth")

model.load_state_dict(
    checkpoint["model"]
)

optimizer.load_state_dict(
    checkpoint["optimizer"]
)
```

This allows training to resume from the last checkpoint.

---

# Common Mistakes

❌ Forgetting `optimizer.step()`

❌ Forgetting `optimizer.zero_grad()`

❌ Using an extremely high learning rate

❌ Choosing the wrong optimizer without experimentation

❌ Not saving optimizer state

---

# Cheat Sheet

| Optimizer | Best For |
|------------|----------|
| SGD | Simple Models |
| SGD + Momentum | CNNs |
| RMSProp | Sequential Models |
| Adam | General Deep Learning |
| AdamW | Transformers & LLMs |

---

# Summary

- Optimizers update model parameters using gradients.
- Gradient Descent is the foundation of optimization.
- Learning Rate controls the step size during updates.
- SGD is simple and memory-efficient.
- Momentum speeds up SGD.
- RMSProp adapts learning rates for each parameter.
- Adam combines Momentum and RMSProp.
- AdamW is the preferred optimizer for modern Transformer and LLM training.
- Proper learning rate selection is often more important than optimizer choice.

---

# 🎤 Interview Questions

1. What is an optimizer?
2. Why do neural networks need optimizers?
3. Explain Gradient Descent.
4. Difference between Batch, Stochastic, and Mini-Batch Gradient Descent.
5. What is Learning Rate?
6. Difference between Adam and AdamW.
7. Why is AdamW preferred for Transformers?
8. What is Weight Decay?
9. Why do we call `optimizer.zero_grad()`?
10. Which optimizer would you choose for fine-tuning a Large Language Model?

---

# 📝 Exercises

### Exercise 1

Train a Linear Regression model using SGD.

---

### Exercise 2

Train the same model using Adam.

Compare:

- Loss
- Number of epochs
- Convergence speed

---

### Exercise 3

Change the learning rate to

```
0.1

0.01

0.001

0.0001
```

Observe how training changes.

---

### Exercise 4

Save the optimizer state and resume training.

---

### Exercise 5

Research which optimizer is used in:

- GPT-4 (publicly disclosed information is limited)
- Llama
- Gemma
- Qwen
- Stable Diffusion

---



# Module 8 – Datasets & DataLoaders (`torch.utils.data`)

> **Goal:** Learn how PyTorch loads, batches, shuffles, and efficiently feeds data to deep learning models.

---

# 📚 Table of Contents

- Introduction
- Why Data Loading is Important
- What is a Dataset?
- What is a DataLoader?
- Dataset vs DataLoader
- Built-in Datasets
- Custom Dataset
- `__init__()`
- `__len__()`
- `__getitem__()`
- DataLoader
- Batch Size
- Shuffle
- num_workers
- pin_memory
- drop_last
- Collate Function
- Data Augmentation
- Complete Example
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you're training a model on

```
10 Million Images
```

Would you write

```python
image1 = ...

image2 = ...

image3 = ...

...

image10000000 = ...
```

Of course not.

Instead,

we need a system that can

- Read images from disk
- Read CSV files
- Read text
- Read videos
- Read audio

and provide them to the neural network

efficiently.

That system is

```
Dataset

+

DataLoader
```

---

# Why Data Loading Matters

A GPU can perform billions of operations per second.

But if the GPU waits for data,

training becomes slow.

```
GPU Waiting

↓

No Training

↓

Poor Performance
```

The goal is

```
Always Keep GPU Busy
```

PyTorch's DataLoader is designed for this purpose.

---

# Training Pipeline

```
Dataset

↓

DataLoader

↓

Batch

↓

GPU

↓

Model

↓

Loss

↓

Backward

↓

Optimizer
```

---

# What is a Dataset?

A Dataset is simply

> **A collection of training examples.**

Examples

```
Images

CSV Files

Text

Audio

Videos

Medical Records
```

PyTorch represents every dataset using

```
Dataset Class
```

---

# Built-in Datasets

PyTorch already provides

```python
torchvision.datasets
```

Popular datasets

- MNIST
- CIFAR10
- CIFAR100
- FashionMNIST
- ImageNet

Example

```python
from torchvision.datasets import MNIST
```

---

# Dataset vs DataLoader

Dataset

```
Stores Data
```

DataLoader

```
Loads Data Efficiently
```

Example

```
Dataset

↓

Image 1

Image 2

Image 3

...

↓

DataLoader

↓

Batch

↓

Model
```

---

# Custom Dataset

Most real-world datasets are custom.

To create one,

inherit

```python
Dataset
```

Example

```python
from torch.utils.data import Dataset
```

---

# Dataset Structure

Every Dataset needs

```
__init__()

__len__()

__getitem__()
```

---

# __init__()

Runs once.

Used for

- Reading file names
- Loading CSV
- Initializing variables

Example

```python
class MyDataset(Dataset):

    def __init__(self):

        pass
```

---

# __len__()

Returns

```
Total Number

of Samples
```

Example

```python
def __len__(self):

    return len(self.data)
```

---

# __getitem__()

Returns

```
One Sample
```

Example

```python
def __getitem__(self,index):

    return self.data[index]
```

This is the most important method.

---

# Complete Dataset Example

```python
from torch.utils.data import Dataset

class NumberDataset(Dataset):

    def __init__(self):

        self.data=[1,2,3,4,5]

    def __len__(self):

        return len(self.data)

    def __getitem__(self,index):

        return self.data[index]
```

---

# Creating Dataset

```python
dataset = NumberDataset()

print(len(dataset))
```

Output

```
5
```

---

# Accessing Data

```python
print(dataset[0])
```

Output

```
1
```

---

# What is DataLoader?

DataLoader automatically

- Creates batches
- Shuffles data
- Loads multiple samples
- Supports multiprocessing

Instead of

```
One Image

↓

Model
```

we send

```
Batch

↓

Model
```

---

# Creating DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=True
)
```

---

# Batch Size

Batch Size means

```
How many samples

go together
```

Example

Dataset

```
10 Images
```

Batch Size

```
2
```

Output

```
Batch 1

Image1

Image2

---------------

Batch2

Image3

Image4

---------------
```

---

# Iterating DataLoader

```python
for batch in loader:

    print(batch)
```

Possible Output

```
tensor([2,5])

tensor([1,4])

tensor([3])
```

---

# Shuffle

Training

```python
shuffle=True
```

Every epoch

Order changes.

```
Epoch1

1 2 3 4 5

Epoch2

3 5 1 4 2
```

Helps prevent overfitting to data order.

---

# num_workers

Controls

```
How many CPU

Processes

Load Data
```

Example

```python
DataLoader(

dataset,

num_workers=4
)
```

Higher values may improve throughput depending on your hardware.

---

# pin_memory

Useful for GPU training.

```python
DataLoader(

dataset,

pin_memory=True
)
```

Pinned memory can make CPU-to-GPU transfers more efficient.

---

# drop_last

Suppose

```
10 Samples

Batch Size=3
```

Output

```
3

3

3

1
```

Last Batch

Only

```
1 Sample
```

If

```python
drop_last=True
```

Last incomplete batch is discarded.

---

# Complete DataLoader

```python
loader = DataLoader(

dataset,

batch_size=32,

shuffle=True,

num_workers=4,

pin_memory=True
)
```

---

# Multiple Inputs

Dataset can return

```python
Image

Label
```

Example

```python
return image,label
```

Training

```python
for images,labels in loader:

    print(images.shape)

    print(labels.shape)
```

---

# Custom CSV Dataset

```python
import pandas as pd

class CSVDataset(Dataset):

    def __init__(self,file):

        self.df=pd.read_csv(file)

    def __len__(self):

        return len(self.df)

    def __getitem__(self,index):

        x=self.df.iloc[index,:-1].values

        y=self.df.iloc[index,-1]

        return x,y
```

---

# Image Dataset

```python
from torchvision.io import read_image

image = read_image(path)
```

Usually combined with

```
Transforms
```

such as resizing and normalization.

---

# Data Augmentation

Used to increase dataset diversity.

Examples

- Flip
- Rotate
- Crop
- Resize
- Normalize
- Color Jitter

```python
from torchvision import transforms

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor()

])
```

---

# Collate Function

Default DataLoader stacks tensors into batches.

Sometimes

data has

```
Different Lengths
```

Example

- Sentences
- Documents
- Audio

A custom `collate_fn` lets you control how samples are combined into a batch.

```python
loader = DataLoader(

dataset,

collate_fn=my_collate
)
```

This is especially important in NLP and LLM training.

---

# Complete Training Example

```python
loader = DataLoader(

dataset,

batch_size=32,

shuffle=True
)

for images,labels in loader:

    outputs=model(images)

    loss=criterion(outputs,labels)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

---

# DataLoader Workflow

```
CSV

↓

Dataset

↓

DataLoader

↓

Batch

↓

GPU

↓

Model

↓

Prediction
```

---

# Best Practices

✅ Use DataLoader instead of manual loops.

✅ Shuffle training data.

✅ Use appropriate batch size.

✅ Increase `num_workers` after testing performance.

✅ Use `pin_memory=True` when training on CUDA.

✅ Keep preprocessing inside Dataset or transforms.

---

# Common Mistakes

❌ Forgetting `shuffle=True` for training.

❌ Loading the entire dataset into GPU memory.

❌ Choosing a batch size larger than GPU memory allows.

❌ Performing expensive preprocessing inside the training loop instead of in the Dataset or transforms.

---

# Cheat Sheet

| Component | Purpose |
|------------|---------|
| Dataset | Stores data |
| DataLoader | Loads batches |
| `batch_size` | Samples per batch |
| `shuffle` | Randomizes order |
| `num_workers` | Parallel data loading |
| `pin_memory` | Faster CPU → GPU transfer |
| `drop_last` | Drops incomplete final batch |
| `collate_fn` | Custom batch creation |

---

# Summary

- Dataset represents a collection of samples.
- DataLoader efficiently feeds batches to the model.
- Every custom Dataset implements `__init__()`, `__len__()`, and `__getitem__()`.
- Batching improves GPU utilization.
- Shuffling helps models generalize better.
- `num_workers` enables parallel data loading.
- `pin_memory` can speed up GPU transfers.
- `collate_fn` is useful for variable-length data such as text.

---

# 🎤 Interview Questions

1. What is the difference between Dataset and DataLoader?
2. Why do we need `__getitem__()`?
3. What does `__len__()` return?
4. Why do we use batching?
5. What is `shuffle=True` used for?
6. What does `num_workers` do?
7. What is the purpose of `pin_memory`?
8. What does `drop_last=True` do?
9. When would you write a custom `collate_fn`?
10. Why shouldn't we load an entire large dataset into GPU memory?

---

# 📝 Exercises

### Exercise 1

Create a custom Dataset containing numbers from 1 to 100.

---

### Exercise 2

Create a DataLoader with

```
batch_size = 10
```

and print each batch.

---

### Exercise 3

Load a CSV file using a custom Dataset.

---

### Exercise 4

Create an image Dataset using `torchvision`.

---

### Exercise 5

Experiment with different values of

- `batch_size`
- `shuffle`
- `num_workers`

Observe how they affect training speed.

---

# Module 9 – Training Loop

> **Goal:** Learn how to train a neural network from scratch using PyTorch by combining tensors, models, loss functions, optimizers, and datasets into a complete training pipeline.

---

# 📚 Table of Contents

- Introduction
- What is a Training Loop?
- Why Do We Need a Training Loop?
- Complete Training Pipeline
- Epoch
- Batch
- Iteration
- Forward Pass
- Loss Calculation
- Backward Pass
- Optimizer Step
- Zero Gradients
- Model Training
- Logging Loss
- Validation Loop
- Saving Model
- Complete Example
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you have a student preparing for an exam.

The learning process looks like this:

```
Read Notes

↓

Attempt Questions

↓

Check Answers

↓

Find Mistakes

↓

Learn From Mistakes

↓

Repeat
```

A neural network learns in exactly the same way.

Instead of notes, it receives **training data**.

Instead of checking answers manually, it computes a **loss**.

Instead of remembering mistakes, it updates its **weights**.

This repeated learning process is called the **Training Loop**.

---

# What is a Training Loop?

A training loop is the repeated process of:

1. Reading a batch of data
2. Making predictions
3. Computing loss
4. Calculating gradients
5. Updating model weights
6. Repeating until the model learns

Without a training loop, a neural network cannot improve.

---

# Complete Training Pipeline

```text
Dataset

↓

DataLoader

↓

Batch

↓

Model

↓

Prediction

↓

Loss Function

↓

Backward()

↓

Optimizer

↓

Updated Weights

↓

Next Batch
```

---

# Important Terms

## Epoch

An **Epoch** means

> One complete pass through the entire training dataset.

Example

```
1000 Images

↓

Model sees all 1000 images once

=

1 Epoch
```

---

## Batch

A **Batch** is a small subset of the dataset.

Example

```
Dataset

1000 Images

Batch Size = 100

↓

10 Batches
```

---

## Iteration

One iteration means processing **one batch**.

Example

```
Epoch = 1

Batch Size = 100

Dataset = 1000 Images

Iterations = 10
```

Relationship

```
Dataset

↓

Epoch

↓

Batches

↓

Iterations
```

---

# Forward Pass

The input data is passed through the neural network.

```text
Input

↓

Model

↓

Prediction
```

PyTorch

```python
outputs = model(images)
```

---

# Loss Calculation

Compare prediction with actual labels.

```python
loss = criterion(outputs, labels)
```

Smaller loss

↓

Better model

---

# Backward Pass

Compute gradients.

```python
loss.backward()
```

Autograd calculates

```
∂Loss/∂Weight
```

for every trainable parameter.

---

# Optimizer Step

Update weights.

```python
optimizer.step()
```

Formula

```
New Weight

=

Old Weight

-

Learning Rate

×

Gradient
```

---

# Zero Gradients

Reset gradients before the next iteration.

```python
optimizer.zero_grad()
```

Why?

Because gradients accumulate by default.

---

# Order of Operations

Always follow this sequence:

```text
Forward Pass

↓

Loss Calculation

↓

Zero Gradients

↓

Backward Pass

↓

Optimizer Step
```

Common implementation:

```python
optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()
```

---

# Training Loop Skeleton

```python
for epoch in range(num_epochs):

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
```

This pattern appears in almost every PyTorch project.

---

# Logging Loss

Monitor training progress.

```python
print(loss.item())
```

Example

```
Epoch 1

Loss = 2.31

Epoch 2

Loss = 1.65

Epoch 3

Loss = 0.82
```

A decreasing loss generally indicates learning.

---

# Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = SimpleNN()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch+1}, Loss: {loss.item():.4f}"
    )
```

---

# Training on GPU

```python
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

model = model.to(device)

for images, labels in train_loader:

    images = images.to(device)

    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()
```

---

# Validation Loop

Training and validation are different.

Training

```
Weights Update

YES
```

Validation

```
Weights Update

NO
```

Example

```python
model.eval()

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)
```

---

# Why `model.train()`?

```python
model.train()
```

Enables training behavior for layers like:

- Dropout
- BatchNorm

---

# Why `model.eval()`?

```python
model.eval()
```

Switches the model to inference mode.

Required before validation or testing.

---

# Saving the Model

```python
torch.save(
    model.state_dict(),
    "model.pth"
)
```

---

# Loading the Model

```python
model.load_state_dict(
    torch.load("model.pth")
)

model.eval()
```

---

# Saving Checkpoints

```python
torch.save({

    "epoch": epoch,

    "model": model.state_dict(),

    "optimizer": optimizer.state_dict()

}, "checkpoint.pth")
```

Useful for resuming training.

---

# Visual Workflow

```text
Data

↓

Forward Pass

↓

Prediction

↓

Loss

↓

Backward

↓

Gradients

↓

Optimizer

↓

Updated Model

↓

Repeat
```

---

# Best Practices

✅ Shuffle training data.

✅ Move data and model to the same device.

✅ Use `optimizer.zero_grad()` every iteration.

✅ Call `model.train()` before training.

✅ Call `model.eval()` before validation.

✅ Save checkpoints regularly.

---

# Common Mistakes

❌ Forgetting `optimizer.zero_grad()`.

❌ Forgetting `model.train()`.

❌ Running validation without `torch.no_grad()`.

❌ Mixing CPU and GPU tensors.

❌ Saving the entire model instead of `state_dict()`.

---

# Cheat Sheet

| Step | Code |
|------|------|
| Training Mode | `model.train()` |
| Forward Pass | `outputs = model(images)` |
| Compute Loss | `loss = criterion(outputs, labels)` |
| Clear Gradients | `optimizer.zero_grad()` |
| Backward Pass | `loss.backward()` |
| Update Weights | `optimizer.step()` |
| Evaluation Mode | `model.eval()` |
| Disable Gradients | `with torch.no_grad():` |
| Save Model | `torch.save(model.state_dict(), "model.pth")` |

---

# Summary

- A training loop is the core of every deep learning model.
- Training consists of forward pass, loss calculation, backward pass, and optimizer step.
- Epochs, batches, and iterations define how data is processed.
- Validation uses `model.eval()` and `torch.no_grad()`.
- Saving checkpoints enables training to resume later.
- Following the correct order of operations ensures stable learning.

---

# 🎤 Interview Questions

1. What is a training loop?
2. Difference between epoch, batch, and iteration?
3. Why do we call `optimizer.zero_grad()`?
4. What does `loss.backward()` do?
5. Why is `optimizer.step()` necessary?
6. Difference between `model.train()` and `model.eval()`?
7. Why is `torch.no_grad()` used during validation?
8. Why save `state_dict()` instead of the whole model?
9. What should happen if the loss is not decreasing?
10. How would you resume training from a checkpoint?

---

# 📝 Exercises

### Exercise 1

Write a training loop for a Linear Regression model.

---

### Exercise 2

Train an MNIST classifier for 5 epochs.

---

### Exercise 3

Print the loss after every batch and every epoch.

---

### Exercise 4

Add validation after each epoch using `model.eval()`.

---

### Exercise 5

Save the best model checkpoint based on validation loss.

---

# Module 10 – Validation, Testing & Model Evaluation

> **Goal:** Learn how to properly evaluate deep learning models, understand different evaluation metrics, avoid overfitting, and choose the best model for deployment.

---

# 📚 Table of Contents

- Introduction
- Why Model Evaluation?
- Training vs Validation vs Testing
- Overfitting & Underfitting
- Validation Pipeline
- model.train() vs model.eval()
- torch.no_grad()
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC
- Validation Loss
- Early Stopping
- Saving Best Model
- Complete Validation Loop
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you are preparing for an exam.

You solve

```
100 Practice Questions
```

every day.

Eventually,

you memorize every question.

Now,

your friend gives you

```
20 Completely New Questions
```

If you score well,

it means

```
You Learned
```

If you score poorly,

it means

```
You Memorized
```

Deep Learning models behave exactly the same way.

A model should **learn patterns**, not memorize the training data.

To check this,

we evaluate the model on **unseen data**.

---

# Why Do We Need Validation?

Suppose

Training Accuracy

```
100%
```

Looks amazing.

But

Testing Accuracy

```
45%
```

Something is wrong.

The model memorized

instead of

learning.

This problem is called

```
Overfitting
```

Validation helps us detect it early.

---

# Dataset Split

A dataset is usually divided into

```
100%

↓

Training

↓

Validation

↓

Testing
```

Typical split

```
70%

↓

Training

20%

↓

Validation

10%

↓

Testing
```

or

```
80%

10%

10%
```

---

# Purpose of Each Dataset

## Training Set

Used to

```
Learn
```

Weights are updated.

---

## Validation Set

Used to

```
Tune

Hyperparameters
```

Weights are NOT updated.

---

## Test Set

Used only once.

Purpose

```
Final Performance
```

Never train on test data.

---

# Complete Pipeline

```
Training Data

↓

Train Model

↓

Validation Data

↓

Choose Best Model

↓

Test Data

↓

Final Accuracy
```

---

# model.train()

Training Mode

```python
model.train()
```

Enables

- Dropout
- BatchNorm Updates

Used only during training.

---

# model.eval()

Evaluation Mode

```python
model.eval()
```

Disables

- Dropout randomness
- BatchNorm updates

Always use

```
model.eval()
```

during validation and testing.

---

# torch.no_grad()

Validation doesn't require gradients.

Example

```python
with torch.no_grad():

    outputs = model(images)
```

Advantages

✅ Faster

✅ Less Memory

✅ No Gradient Computation

---

# Validation Loop

```python
model.eval()

total_loss = 0

correct = 0

total = 0

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

accuracy = correct / total

print("Validation Accuracy:", accuracy)
```

---

# Training vs Validation

| Feature | Training | Validation |
|----------|-----------|------------|
| Updates Weights | ✅ | ❌ |
| Computes Gradients | ✅ | ❌ |
| Uses Backward | ✅ | ❌ |
| Uses Optimizer | ✅ | ❌ |
| model.train() | ✅ | ❌ |
| model.eval() | ❌ | ✅ |

---

# Accuracy

Most common metric.

Formula

```
Correct Predictions

/

Total Predictions
```

Example

```
100 Images

↓

95 Correct

↓

Accuracy

95%
```

---

# PyTorch Example

```python
predictions = outputs.argmax(dim=1)

accuracy = (

predictions == labels

).float().mean()
```

---

# Precision

Precision answers

```
Among

all predicted positives,

how many

were actually positive?
```

Formula

```
TP

/

TP+FP
```

Useful for

- Spam Detection
- Medical Diagnosis

---

# Recall

Recall answers

```
Among

all actual positives,

how many

did we correctly find?
```

Formula

```
TP

/

TP+FN
```

Very important when

missing positives

is expensive.

Example

Cancer Detection.

---

# Precision vs Recall

Imagine

100 Emails

Spam

```
20
```

Your model marks

```
15

as Spam
```

Precision

↓

How many predicted spam emails were actually spam?

Recall

↓

How many of the real spam emails did we detect?

---

# F1 Score

Sometimes

Accuracy is misleading.

F1 combines

```
Precision

+

Recall
```

Formula

```
2PR

/

P+R
```

Used for

- Imbalanced Data
- Fraud Detection
- Medical AI

---

# Confusion Matrix

Shows

Correct

and

Wrong Predictions.

```
                 Predicted

            Yes        No

Actual Yes   TP        FN

Actual No    FP        TN
```

Where

TP

True Positive

FP

False Positive

FN

False Negative

TN

True Negative

---

# ROC Curve

ROC

stands for

```
Receiver Operating Characteristic
```

Shows

```
True Positive Rate

vs

False Positive Rate
```

A good classifier

has a curve

closer to

top-left.

---

# AUC

Area Under Curve

Range

```
0

↓

1
```

```
1

Perfect Model
```

```
0.5

Random Guess
```

Higher AUC

↓

Better Model

---

# Validation Loss

Don't only monitor

Accuracy.

Also monitor

```
Validation Loss
```

Sometimes

Accuracy

stays same

while

Loss

increases.

This may indicate

overfitting.

---

# Overfitting

```
Training Accuracy

99%

↓

Validation Accuracy

70%
```

Model memorized

training data.

---

# Underfitting

```
Training Accuracy

55%

↓

Validation Accuracy

53%
```

Model

didn't learn enough.

---

# Ideal Model

```
Training

95%

↓

Validation

94%
```

Small difference.

Good Generalization.

---

# Early Stopping

Instead of

training

100 epochs,

stop when

validation loss

stops improving.

Pseudo Logic

```
Validation Loss

↓

Improving

Continue

↓

No Improvement

Stop
```

Advantages

- Saves Time
- Prevents Overfitting

---

# Save Best Model

```python
best_loss = float("inf")

if val_loss < best_loss:

    best_loss = val_loss

    torch.save(

        model.state_dict(),

        "best_model.pth"
    )
```

Never save only

the last epoch.

Save

the

best model.

---

# Visual Workflow

```
Training

↓

Validation

↓

Metrics

↓

Choose Best Model

↓

Testing

↓

Deployment
```

---

# Common Evaluation Metrics

| Metric | Used For |
|----------|-----------|
| Accuracy | General Classification |
| Precision | Spam Detection |
| Recall | Medical Diagnosis |
| F1 Score | Imbalanced Dataset |
| ROC-AUC | Binary Classification |

---

# Best Practices

✅ Use separate Validation and Test datasets.

✅ Always call

```python
model.eval()
```

during evaluation.

✅ Use

```python
torch.no_grad()
```

for validation.

✅ Save the best model.

✅ Monitor both

Accuracy

and

Loss.

---

# Common Mistakes

❌ Evaluating in

```
model.train()
```

mode.

---

❌ Forgetting

```
torch.no_grad()
```

---

❌ Using

Test Data

for Hyperparameter Tuning.

---

❌ Looking only at

Accuracy.

---

# Cheat Sheet

| Task | Code |
|------|------|
| Training Mode | `model.train()` |
| Evaluation Mode | `model.eval()` |
| Disable Gradients | `with torch.no_grad():` |
| Prediction | `outputs.argmax(dim=1)` |
| Save Best Model | `torch.save()` |

---

# Summary

- Validation measures how well the model generalizes to unseen data.
- Testing is used only for the final evaluation.
- `model.train()` and `model.eval()` control the behavior of certain layers.
- `torch.no_grad()` speeds up inference and reduces memory usage.
- Accuracy alone is not enough—precision, recall, F1 score, and ROC-AUC provide deeper insights.
- Early stopping helps prevent overfitting.
- Save the best-performing model based on validation performance.

---

# 🎤 Interview Questions

1. Difference between Training, Validation, and Testing?
2. Why use `model.eval()`?
3. Why use `torch.no_grad()`?
4. Difference between Precision and Recall?
5. What is F1 Score?
6. What is a Confusion Matrix?
7. What is ROC-AUC?
8. What is Overfitting?
9. What is Underfitting?
10. Why should we save the best model instead of the last model?

---

# 📝 Exercises

### Exercise 1

Implement a validation loop.

---

### Exercise 2

Compute validation accuracy after every epoch.

---

### Exercise 3

Save the best model using validation loss.

---

### Exercise 4

Research Precision, Recall, and F1 Score for an imbalanced dataset.

---

### Exercise 5

Create a confusion matrix for a simple classifier.

---

# Module 11 – Saving & Loading Models

> **Goal:** Learn how to save, load, resume, and deploy PyTorch models using best practices followed in real-world AI and LLM projects.

---

# 📚 Table of Contents

- Introduction
- Why Save Models?
- What is state_dict()?
- Saving Model Weights
- Loading Model Weights
- Saving Entire Model
- Loading Entire Model
- Saving Checkpoints
- Resume Training
- Saving Optimizer State
- Saving Multiple Models
- Model Versioning
- Inference Workflow
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you are training

```
Llama

7 Billion Parameters
```

Training takes

```
4 Days
```

After

```
3 Days

23 Hours
```

Power goes off.

Computer crashes.

Everything is lost.

Would you start again?

```
No.
```

Instead,

every few minutes

or every epoch,

you save

```
Model

Optimizer

Epoch

Loss
```

This saved information is called a

```
Checkpoint
```

Checkpointing is one of the most important concepts in Deep Learning.

---

# Why Save Models?

Training takes

```
Minutes

Hours

Days

Weeks
```

Examples

| Model | Training Time |
|--------|---------------|
| CNN | Minutes |
| ResNet | Hours |
| ViT | Days |
| Llama | Weeks |
| GPT | Months |

Without saving,

all progress is lost.

---

# What is state_dict()?

Every PyTorch model contains

```
Weights

Biases
```

PyTorch stores them inside

```
state_dict()
```

Think of it as

```
Dictionary

↓

Parameter Name

↓

Tensor
```

---

# Example

```python
for key, value in model.state_dict().items():

    print(key)
```

Output

```
fc1.weight

fc1.bias

fc2.weight

fc2.bias
```

---

# Why state_dict()?

Instead of saving

```
Entire Model
```

PyTorch recommends saving only

```
Weights
```

Advantages

✅ Smaller File

✅ Portable

✅ Faster

✅ Recommended by PyTorch

---

# Saving Model

Syntax

```python
torch.save(
    model.state_dict(),
    "model.pth"
)
```

Example

```python
torch.save(

    model.state_dict(),

    "best_model.pth"
)
```

Output

```
best_model.pth
```

---

# What is .pth?

`.pth`

stands for

```
PyTorch File
```

Common extensions

```
.pth

.pt

.ckpt
```

All are commonly used in PyTorch projects.

---

# Loading Model

Before loading,

first create

the model.

```python
model = SimpleNN()
```

Then

```python
model.load_state_dict(

    torch.load("best_model.pth")
)
```

Finally

```python
model.eval()
```

---

# Why model.eval()?

Loading weights

doesn't automatically

switch

to inference mode.

Always call

```python
model.eval()
```

before prediction.

---

# Saving Entire Model

Possible

```python
torch.save(

    model,

    "model.pth"
)
```

Loading

```python
model = torch.load(

    "model.pth"
)
```

---

# Why Not Save Entire Model?

Because

Entire Model

contains

```
Python Class

File Structure

Dependencies
```

Changing code later

may break loading.

Therefore

PyTorch recommends

```
state_dict()
```

---

# Saving Checkpoint

Real projects

save

```
Epoch

Model

Optimizer

Loss
```

Example

```python
torch.save({

    "epoch": epoch,

    "model_state_dict":
    model.state_dict(),

    "optimizer_state_dict":
    optimizer.state_dict(),

    "loss": loss

},

"checkpoint.pth")
```

---

# Loading Checkpoint

```python
checkpoint = torch.load(

    "checkpoint.pth"
)

model.load_state_dict(

    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(

    checkpoint["optimizer_state_dict"]
)

epoch = checkpoint["epoch"]

loss = checkpoint["loss"]
```

Now

training continues

from the saved epoch.

---

# Resume Training

```python
for epoch in range(

checkpoint["epoch"],

100

):
```

Instead of

starting

from

```
Epoch 0
```

training resumes

from

```
Saved Epoch
```

---

# Saving Best Model

Suppose

Validation Loss

```
0.35

↓

0.31

↓

0.28

↓

0.30
```

Best model

is

```
0.28
```

Example

```python
if val_loss < best_loss:

    best_loss = val_loss

    torch.save(

        model.state_dict(),

        "best_model.pth"
    )
```

---

# Saving Optimizer

Optimizer

contains

```
Momentum

Learning Rate

Internal State
```

Always save

```python
optimizer.state_dict()
```

Otherwise

training won't continue

exactly where it stopped.

---

# Saving Multiple Models

GAN Example

```
Generator

↓

Save

Discriminator

↓

Save
```

```python
torch.save({

"generator":

generator.state_dict(),

"discriminator":

discriminator.state_dict()

},

"gan.pth")
```

---

# Saving Scheduler

If using

Learning Rate Scheduler

also save

```python
scheduler.state_dict()
```

---

# File Structure

```
Project

│

├── checkpoints/

│      epoch10.pth

│      epoch20.pth

│

├── best_model.pth

│

├── latest_model.pth

│

└── train.py
```

---

# Inference Workflow

```
Load Model

↓

Load Weights

↓

model.eval()

↓

torch.no_grad()

↓

Prediction
```

Example

```python
model.load_state_dict(

torch.load("best_model.pth")
)

model.eval()

with torch.no_grad():

    prediction = model(image)
```

---

# Saving on GPU

If model

trained on GPU

```python
torch.save(

model.state_dict(),

"gpu_model.pth"
)
```

Loading on CPU

```python
torch.load(

"gpu_model.pth",

map_location="cpu"
)
```

Very useful

when deployment

doesn't have GPU.

---

# Save Checkpoint Every Epoch

```python
torch.save(

model.state_dict(),

f"epoch_{epoch}.pth"
)
```

Output

```
epoch_1.pth

epoch_2.pth

epoch_3.pth
```

---

# Common File Extensions

| Extension | Purpose |
|------------|----------|
| .pth | Model Weights |
| .pt | Model File |
| .ckpt | Checkpoint |
| .bin | Hugging Face Models |
| .safetensors | Modern LLM Models |

---

# Relation with LLMs

PyTorch

```
↓

state_dict()

↓

Weights

↓

.safetensors

↓

Hugging Face
```

When you fine-tune

```
Llama

Gemma

Qwen
```

their weights are commonly saved as

```
.safetensors
```

because the format is designed to safely and efficiently store tensors.

---

# Best Practices

✅ Save only

```
state_dict()
```

---

✅ Save

Best Model.

---

✅ Save Optimizer.

---

✅ Save Scheduler.

---

✅ Save Epoch Number.

---

✅ Save Validation Loss.

---

# Common Mistakes

❌ Saving

Entire Model

instead of

```
state_dict()
```

---

❌ Forgetting

```python
model.eval()
```

---

❌ Forgetting

Optimizer State.

---

❌ Not Saving

Best Model.

---

❌ Overwriting

Checkpoints.

---

# Cheat Sheet

| Task | Code |
|------|------|
| Save Model | `torch.save(model.state_dict(),"model.pth")` |
| Load Model | `model.load_state_dict(torch.load("model.pth"))` |
| Save Optimizer | `optimizer.state_dict()` |
| Save Checkpoint | `torch.save({...})` |
| Resume Training | `optimizer.load_state_dict()` |
| Inference | `model.eval()` |
| Disable Gradients | `torch.no_grad()` |
| GPU → CPU | `map_location="cpu"` |

---

# Summary

- PyTorch recommends saving only the model's `state_dict()`.
- Checkpoints should include the model, optimizer, epoch, and loss.
- Use `model.eval()` before inference.
- `torch.load(..., map_location="cpu")` helps load GPU-trained models on CPU-only machines.
- Saving checkpoints regularly prevents losing training progress.
- Modern LLMs commonly use formats such as `.safetensors` to store model weights.

---

# 🎤 Interview Questions

1. What is `state_dict()`?
2. Why does PyTorch recommend saving `state_dict()` instead of the whole model?
3. What should a checkpoint contain?
4. Difference between `.pth` and `.safetensors`?
5. Why call `model.eval()` after loading a model?
6. How do you resume training from a checkpoint?
7. What is `map_location="cpu"` used for?
8. Why save the optimizer state?
9. How do you save the best model during training?
10. Why are checkpoints important for large models?

---

# 📝 Exercises

### Exercise 1

Save a trained model using `state_dict()`.

---

### Exercise 2

Load the saved model and perform inference.

---

### Exercise 3

Create a complete training checkpoint containing:

- Model
- Optimizer
- Epoch
- Loss

---

### Exercise 4

Resume training from the saved checkpoint.

---

### Exercise 5

Train a model on GPU (if available) and load it on CPU using `map_location="cpu"`.

---

# Module 12 – Learning Rate Scheduling (`torch.optim.lr_scheduler`)

> **Goal:** Learn how Learning Rate Schedulers improve model training by automatically adjusting the learning rate during training.

---

# 📚 Table of Contents

- Introduction
- Why Learning Rate Matters
- What is a Learning Rate Scheduler?
- Why Not Use a Constant Learning Rate?
- Types of Learning Rate Schedulers
- StepLR
- MultiStepLR
- ExponentialLR
- CosineAnnealingLR
- ReduceLROnPlateau
- CosineAnnealingWarmRestarts
- LinearLR
- SequentialLR
- OneCycleLR
- Warmup Scheduling
- Scheduler Workflow
- Complete Training Example
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you are driving from Mumbai to Pune.

When you leave your home,

you drive normally.

```
80 km/h
```

As you reach Pune city,

you slow down.

```
40 km/h
```

Near your destination,

```
20 km/h
```

Why?

Because driving at

```
80 km/h
```

inside the city

would make you miss your destination.

Deep Learning works exactly the same way.

At the beginning,

the model should take

**large steps**

to learn quickly.

Later,

it should take

**small steps**

to fine-tune the weights.

This gradual adjustment is handled by a

> **Learning Rate Scheduler**

---

# What is Learning Rate?

Learning Rate determines

```
How much

weights change

after every update.
```

Formula

```
New Weight

=

Old Weight

-

Learning Rate

×

Gradient
```

Notation

```
lr
```

Example

```python
lr = 0.001
```

---

# What Happens if Learning Rate is Too High?

Suppose

```
Minimum Loss

↓

●
```

Instead of moving slowly,

the optimizer keeps jumping

from one side

to another.

```
●     ●

   X

●     ●
```

Result

❌ Doesn't Converge

---

# Too Small Learning Rate

```
Very Tiny Steps

↓

Very Slow Learning
```

Training may require

thousands of epochs.

---

# Ideal Learning Rate

```
Large Steps

↓

Near Minimum

↓

Small Steps

↓

Converges
```

---

# Why Use Learning Rate Scheduler?

Instead of

```
LR = 0.001

Forever
```

Schedulers automatically change

```
0.001

↓

0.0005

↓

0.0001

↓

0.00001
```

Advantages

✅ Faster Training

✅ Better Accuracy

✅ Stable Optimization

---

# Scheduler Workflow

```
Forward Pass

↓

Loss

↓

Backward

↓

Optimizer Step

↓

Scheduler Step

↓

New Learning Rate
```

---

# Creating Optimizer

```python
import torch.optim as optim

optimizer = optim.Adam(

    model.parameters(),

    lr=0.001
)
```

---

# StepLR

Most basic scheduler.

Reduces learning rate

after every

fixed number

of epochs.

Example

```python
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(

    optimizer,

    step_size=5,

    gamma=0.1
)
```

Meaning

Every

```
5 Epochs
```

Learning Rate becomes

```
Old LR × 0.1
```

---

# Example

Initial LR

```
0.001
```

Epoch

```
1

↓

0.001
```

Epoch

```
5

↓

0.0001
```

Epoch

```
10

↓

0.00001
```

---

# Using StepLR

```python
for epoch in range(20):

    train()

    scheduler.step()
```

---

# MultiStepLR

Instead of

every

5 epochs,

reduce only at

specific epochs.

```python
from torch.optim.lr_scheduler import MultiStepLR

scheduler = MultiStepLR(

optimizer,

milestones=[10,20,30],

gamma=0.1
)
```

Learning Rate changes

only at

```
Epoch

10

20

30
```

---

# ExponentialLR

Reduce

every epoch

using exponential decay.

```python
from torch.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(

optimizer,

gamma=0.95
)
```

Formula

```
LR

↓

LR × 0.95

↓

LR × 0.95

↓

...
```

---

# CosineAnnealingLR

Very popular

for CNNs

and Transformers.

Learning Rate follows

Cosine Curve.

```
LR

|

|\
| \
|  \
|   \
|    \____

+------------

Epoch
```

PyTorch

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(

optimizer,

T_max=20
)
```

Advantages

- Smooth decay
- Better convergence
- Used in modern research

---

# ReduceLROnPlateau

One of the smartest schedulers.

Instead of

reducing every epoch,

it watches

```
Validation Loss
```

If

Validation Loss

doesn't improve,

Learning Rate decreases.

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(

optimizer,

mode="min",

factor=0.1,

patience=3
)
```

Meaning

If validation loss

doesn't improve

for

```
3 Epochs
```

↓

Reduce LR.

---

# Usage

```python
scheduler.step(

validation_loss
)
```

Unlike most schedulers,

this one receives

the validation metric.

---

# CosineAnnealingWarmRestarts

Instead of continuously decreasing,

learning rate periodically resets.

```
LR

|\

| \

|  \

|   \

|    /\

|   /  \

|__/____\____
```

Useful

for

long training.

---

# LinearLR

Changes

Learning Rate

linearly.

```python
scheduler = torch.optim.lr_scheduler.LinearLR(

optimizer,

start_factor=0.5,

total_iters=5
)
```

Often used during

Warmup.

---

# OneCycleLR

One of the most effective schedulers.

Learning Rate

```
Increase

↓

Peak

↓

Decrease
```

PyTorch

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(

optimizer,

max_lr=0.01,

epochs=10,

steps_per_epoch=len(train_loader)
)
```

Popular

for

CNN

Vision

Classification

---

# Warmup Scheduler

Large Transformers

usually don't start with

high learning rate.

Instead

```
0

↓

Small

↓

Medium

↓

Maximum

↓

Decay
```

Called

```
Learning Rate Warmup
```

Why?

Prevents unstable updates

during the initial training steps.

Common in

- BERT
- GPT
- Llama
- Gemma
- Qwen

---

# SequentialLR

Combine

multiple schedulers.

Example

```
Warmup

↓

Cosine Decay
```

```python
scheduler = torch.optim.lr_scheduler.SequentialLR(

optimizer,

schedulers=[warmup, cosine],

milestones=[5]
)
```

---

# Complete Training Example

```python
optimizer = torch.optim.Adam(

model.parameters(),

lr=0.001
)

scheduler = StepLR(

optimizer,

step_size=5,

gamma=0.1
)

for epoch in range(20):

    model.train()

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    scheduler.step()

    print(

    optimizer.param_groups[0]["lr"]

    )
```

---

# Visual Flow

```
Forward

↓

Loss

↓

Backward

↓

Optimizer

↓

Scheduler

↓

Next Epoch
```

---

# Checking Current Learning Rate

```python
print(

optimizer.param_groups[0]["lr"]

)
```

Output

```
0.001
```

---

# Optimizer vs Scheduler

| Optimizer | Scheduler |
|------------|------------|
| Updates Weights | Updates Learning Rate |
| Uses Gradients | Uses Epoch or Metric |
| Every Iteration | Every Epoch (usually) |

---

# Which Scheduler Should I Use?

| Problem | Scheduler |
|----------|------------|
| Beginner Projects | StepLR |
| CNN | CosineAnnealingLR |
| Vision Transformer | CosineAnnealingLR |
| Transformer | Warmup + Cosine |
| LLM Training | Warmup + Cosine |
| Validation Based | ReduceLROnPlateau |

---

# Real World Usage

| Model | Scheduler |
|---------|------------|
| ResNet | StepLR / Cosine |
| EfficientNet | Cosine |
| ViT | Cosine |
| BERT | Linear Warmup |
| GPT | Cosine + Warmup |
| Llama | Cosine + Warmup |
| Gemma | Cosine + Warmup |
| Qwen | Cosine + Warmup |

---

# Best Practices

✅ Monitor Learning Rate.

✅ Don't use

very large

Learning Rate.

✅ Use Warmup

for Transformers.

✅ Use Cosine Scheduler

for Deep Learning.

✅ Save Scheduler

inside checkpoint.

---

# Saving Scheduler

```python
torch.save({

"model":

model.state_dict(),

"optimizer":

optimizer.state_dict(),

"scheduler":

scheduler.state_dict()

},

"checkpoint.pth")
```

Loading

```python
scheduler.load_state_dict(

checkpoint["scheduler"]
)
```

---

# Common Mistakes

❌ Forgetting

```python
scheduler.step()
```

---

❌ Calling scheduler

before optimizer

(for schedulers designed to step after optimizer updates or after each epoch).

---

❌ Using

ReduceLROnPlateau

without passing

validation loss.

---

❌ Forgetting

to save scheduler state.

---

# Cheat Sheet

| Scheduler | Best Use |
|------------|-----------|
| StepLR | Simple Projects |
| MultiStepLR | Fixed Milestones |
| ExponentialLR | Smooth Decay |
| CosineAnnealingLR | CNN, ViT |
| ReduceLROnPlateau | Validation Based |
| OneCycleLR | Fast Training |
| Warmup | Transformers |
| SequentialLR | Combine Multiple Schedulers |

---

# Summary

- Learning Rate controls the step size during optimization.
- Learning Rate Schedulers automatically adjust the learning rate.
- StepLR decreases the learning rate at fixed intervals.
- MultiStepLR changes the learning rate at specific milestones.
- CosineAnnealingLR provides smooth decay and is widely used in modern deep learning.
- ReduceLROnPlateau reacts to validation performance.
- Warmup is commonly used for Transformer and LLM training.
- Saving the scheduler state is important when resuming training.

---

# 🎤 Interview Questions

1. What is a Learning Rate Scheduler?

2. Why not keep the Learning Rate constant?

3. Difference between StepLR and MultiStepLR?

4. When should you use ReduceLROnPlateau?

5. What is CosineAnnealingLR?

6. Why is Warmup important for Transformers?

7. Difference between Optimizer and Scheduler?

8. Why save the Scheduler state?

9. What is OneCycleLR?

10. Which scheduler is commonly used for LLM training?

---

# 📝 Exercises

### Exercise 1

Train a model using

```
StepLR
```

---

### Exercise 2

Replace

```
StepLR

↓

CosineAnnealingLR
```

Compare

- Training Loss

- Validation Loss

---

### Exercise 3

Implement

```
ReduceLROnPlateau
```

using

Validation Loss.

---

### Exercise 4

Print

Learning Rate

after every epoch.

---

### Exercise 5

Save

Optimizer

+

Scheduler

+

Model

inside one checkpoint.

---

# Module 13 – Transfer Learning & Fine-Tuning

> **Goal:** Learn how to use pretrained deep learning models, freeze layers, fine-tune networks, and understand how transfer learning is used in modern AI and LLMs.

---

# 📚 Table of Contents

- Introduction
- Why Transfer Learning?
- What is Transfer Learning?
- Why Does It Work?
- Feature Extraction
- Fine-Tuning
- Pretrained Models
- torchvision Models
- Loading Pretrained Models
- Freezing Layers
- Unfreezing Layers
- Replacing the Classification Head
- Fine-Tuning Workflow
- Practical Example
- Transfer Learning in NLP
- Transfer Learning in LLMs
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine you want to learn

```
Spanish
```

but you already know

```
English
```

Learning Spanish becomes much easier because

you already understand

- Grammar
- Sentence Structure
- Vocabulary Concepts

You don't start from zero.

Instead,

you reuse your previous knowledge.

Deep Learning works the same way.

Instead of training a model from scratch,

we reuse knowledge from a model

already trained on millions of images.

This idea is called

> **Transfer Learning**

---

# What is Transfer Learning?

Transfer Learning means

> Using knowledge learned from one task to solve another task.

Example

```
ImageNet

↓

ResNet

↓

Your Cat vs Dog Dataset
```

Instead of training

```
25 Million Parameters
```

from scratch,

we only train

a few layers.

---

# Why Do We Need Transfer Learning?

Training a Deep Learning model from scratch requires

- Huge datasets
- Powerful GPUs
- Long training time

Example

Training

```
ResNet-50

↓

ImageNet

↓

Millions of Images

↓

Days of Training
```

Instead,

we download

a pretrained model

and fine-tune it.

---

# What is a Pretrained Model?

A pretrained model is a model

that has already been trained

on a very large dataset.

Examples

```
ResNet

EfficientNet

MobileNet

DenseNet

ViT
```

Most computer vision models

are pretrained on

```
ImageNet
```

which contains

```
14 Million+

Images
```

---

# Why Do Pretrained Models Work?

The first layers of CNNs

learn

```
Edges

↓

Textures

↓

Shapes

↓

Patterns
```

These features are useful

for almost every vision task.

So,

instead of learning

them again,

we reuse them.

---

# Transfer Learning Pipeline

```
Large Dataset

↓

Train Model

↓

Save Weights

↓

New Dataset

↓

Load Weights

↓

Fine-Tune

↓

New Model
```

---

# Two Approaches

Transfer Learning mainly has

```
Feature Extraction

↓

Fine-Tuning
```

---

# Feature Extraction

Only train

the final layer.

Everything else

remains frozen.

```
Pretrained Layers

↓

Frozen

↓

New Classifier

↓

Train
```

Advantages

✅ Faster

✅ Less GPU Memory

✅ Small Dataset Friendly

---

# Fine-Tuning

Instead of training

only the last layer,

we also train

some of the pretrained layers.

```
Pretrained Model

↓

Unfreeze Last Layers

↓

Continue Training
```

Advantages

- Better Accuracy
- Learns Dataset Specific Features

---

# torchvision Models

PyTorch provides

many pretrained models

through

```python
torchvision.models
```

Popular models

- ResNet18
- ResNet50
- VGG16
- AlexNet
- DenseNet
- EfficientNet
- Vision Transformer (ViT)
- MobileNet

---

# Loading a Pretrained Model

```python
from torchvision.models import resnet18

model = resnet18(weights="DEFAULT")
```

PyTorch automatically downloads

the pretrained weights.

---

# Exploring the Model

```python
print(model)
```

This prints

all layers

of the network.

---

# Freezing Layers

Freeze means

```
Don't Update Weights
```

Example

```python
for param in model.parameters():

    param.requires_grad = False
```

Now

Backpropagation

will ignore

these layers.

---

# Checking Frozen Parameters

```python
for name, param in model.named_parameters():

    print(

        name,

        param.requires_grad

    )
```

Output

```
layer1.weight False

layer2.weight False
```

---

# Why Freeze Layers?

Imagine

```
25 Million Parameters
```

Training

all of them

requires

- More Memory
- More Time
- More Data

Freezing

reduces

training cost.

---

# Replace Final Layer

ImageNet

has

```
1000 Classes
```

Suppose

your dataset

has only

```
2 Classes
```

Replace

the final classifier.

Example

```python
import torch.nn as nn

model.fc = nn.Linear(

512,

2

)
```

Now

the network predicts

```
Cat

Dog
```

instead of

1000 ImageNet classes.

---

# Train Only Final Layer

```python
optimizer = torch.optim.Adam(

model.fc.parameters(),

lr=0.001
)
```

Only

the new classifier

learns.

Everything else

remains frozen.

---

# Fine-Tuning Last Layers

Later

we may decide

to train

the final block.

Example

```python
for param in model.layer4.parameters():

    param.requires_grad = True
```

Now

Layer 4

also learns.

---

# Complete Example

```python
from torchvision.models import resnet18

import torch.nn as nn

model = resnet18(weights="DEFAULT")

for param in model.parameters():

    param.requires_grad = False

model.fc = nn.Linear(

512,

2
)

optimizer = torch.optim.Adam(

model.fc.parameters(),

lr=0.001
)
```

---

# Training Workflow

```
Load Model

↓

Freeze Layers

↓

Replace Head

↓

Train Classifier

↓

Evaluate

↓

(Optional)

Fine-Tune More Layers
```

---

# Feature Extraction vs Fine-Tuning

| Feature Extraction | Fine-Tuning |
|--------------------|-------------|
| Freeze Backbone | Train Backbone |
| Fast | Slower |
| Less Data Needed | More Data Needed |
| Lower GPU Usage | Higher GPU Usage |
| Beginner Friendly | Better Accuracy |

---

# Transfer Learning in NLP

Transfer Learning

isn't limited

to Computer Vision.

Example

```
BERT

↓

Fine-Tune

↓

Sentiment Analysis
```

or

```
GPT

↓

Fine-Tune

↓

Chatbot
```

---

# Transfer Learning in LLMs

Modern LLMs

are almost never

trained

from scratch.

Instead

```
Llama

↓

Instruction Dataset

↓

Fine-Tuning

↓

Chat Model
```

or

```
Qwen

↓

Medical Dataset

↓

Fine-Tuning

↓

Medical Assistant
```

This is

Transfer Learning.

---

# Relation with LoRA

Traditional Fine-Tuning

```
Train

Every Parameter
```

LoRA

```
Freeze Model

↓

Train Small Adapters
```

We'll study

LoRA

later

during

LLM Fine-Tuning.

---

# Real World Examples

| Task | Model |
|-------|-------|
| Image Classification | ResNet |
| Face Recognition | EfficientNet |
| Medical Imaging | DenseNet |
| OCR | ViT |
| Sentiment Analysis | BERT |
| Chatbot | Llama |
| Code Generation | Qwen |

---

# Best Practices

✅ Start with pretrained models.

✅ Freeze layers when dataset is small.

✅ Replace the classifier layer.

✅ Use a lower learning rate for fine-tuning.

✅ Unfreeze gradually if needed.

---

# Common Mistakes

❌ Forgetting to replace the final layer.

❌ Training all parameters unnecessarily.

❌ Using a high learning rate during fine-tuning.

❌ Forgetting to freeze pretrained layers.

---

# Cheat Sheet

| Task | Code |
|------|------|
| Load Model | `resnet18(weights="DEFAULT")` |
| Freeze Layers | `param.requires_grad=False` |
| Replace Classifier | `model.fc = nn.Linear(...)` |
| Train Head Only | `optimizer = Adam(model.fc.parameters())` |
| Unfreeze Layers | `param.requires_grad=True` |

---

# Summary

- Transfer Learning reuses knowledge from pretrained models.
- Feature Extraction freezes most of the model and trains only the final layers.
- Fine-Tuning updates some or all pretrained layers.
- Pretrained models reduce training time and data requirements.
- Transfer Learning is used in both Computer Vision and NLP.
- Modern LLM fine-tuning is a form of Transfer Learning.

---

# 🤖 How This Applies to LLMs

Traditional CNN

```
ImageNet

↓

ResNet

↓

Fine-Tune
```

LLMs

```
Pretrained Llama

↓

LoRA

↓

Instruction Dataset

↓

Fine-Tuned Chatbot
```

The idea is exactly the same.

The only difference is

the architecture.

---

# 🎤 Interview Questions

1. What is Transfer Learning?
2. Why do we use pretrained models?
3. Difference between Feature Extraction and Fine-Tuning?
4. What is `requires_grad` used for?
5. Why replace the final classifier?
6. Why use a smaller learning rate for fine-tuning?
7. What is a pretrained model?
8. How is Transfer Learning used in LLMs?
9. What is the difference between Full Fine-Tuning and LoRA?
10. Why are pretrained models preferred over training from scratch?

---

# 📝 Exercises

### Exercise 1

Load

```
ResNet18
```

and print its architecture.

---

### Exercise 2

Freeze all layers.

---

### Exercise 3

Replace the final classification layer.

---

### Exercise 4

Train only the new classifier.

---

### Exercise 5

Unfreeze the final ResNet block and compare the results with feature extraction.

---


# Module 14 – Convolutional Neural Networks (CNNs)

> **Goal:** Learn how Convolutional Neural Networks work from scratch, understand every CNN component mathematically and intuitively, and build your first image classifier using PyTorch.

---

# 📚 Table of Contents

- Introduction
- Why CNN?
- Why Not Fully Connected Networks?
- What is an Image?
- Pixels & Channels
- Feature Extraction
- What is a Convolution?
- Kernel (Filter)
- Feature Map
- Stride
- Padding
- Pooling
- Max Pooling
- Average Pooling
- CNN Architecture
- Building a CNN in PyTorch
- CNN Training Example
- Popular CNN Architectures
- CNN vs Transformer
- Best Practices
- Common Mistakes
- Summary
- Interview Questions
- Exercises

---

# 📖 Story

Imagine someone shows you this image.

🐱 Cat

Do you analyze

```
Every Pixel

at the same time?
```

No.

First,

your eyes notice

```
Edges

↓

Shapes

↓

Eyes

↓

Ears

↓

Face

↓

Entire Cat
```

Humans recognize objects

hierarchically.

CNNs do exactly the same thing.

Instead of memorizing pixels,

they learn

small visual patterns first,

then combine them into

complex objects.

---

# What is an Image?

A computer doesn't see

```
Dog

Cat

Car
```

It only sees

Numbers.

Example

Grayscale Image

```
28 × 28
```

Meaning

```
784 Pixels
```

Each pixel stores

```
0

↓

255
```

Example

```
0

Black
```

```
255

White
```

---

# RGB Images

Color images contain

3 channels.

```
Red

Green

Blue
```

Shape

```
Height

×

Width

×

Channels
```

Example

```
224 × 224 × 3
```

PyTorch stores images as

```
Channels

×

Height

×

Width
```

Example

```
3 × 224 × 224
```

---

# Why Not Use Fully Connected Layers?

Suppose

Input Image

```
224 × 224 × 3
```

Total Pixels

```
224 × 224 × 3

=

150,528
```

A single Linear Layer

```
150528

↓

1000
```

Parameters

```
150 Million+

Weights
```

Problems

❌ Huge Memory

❌ Slow

❌ Overfitting

❌ Ignores Spatial Information

CNN solves all these problems.

---

# What is a Convolution?

Convolution means

> Sliding a small filter over an image to detect patterns.

Example

```
Image

□□□□□□□□

Kernel

■■■
■■■
■■■

↓

Move

↓

Move

↓

Move
```

Instead of looking at the whole image,

CNN looks at

small regions.

---

# What is a Kernel (Filter)?

Kernel

=

Small Matrix

Example

```
3 × 3

5 × 5

7 × 7
```

Example

```
1 0 -1

1 0 -1

1 0 -1
```

This kernel detects

Vertical Edges.

Different kernels learn

- Edges
- Corners
- Curves
- Textures
- Shapes

---

# Feature Map

After convolution,

we get

```
Feature Map
```

Example

```
Image

↓

Convolution

↓

Feature Map
```

The feature map highlights

important regions

of the image.

---

# Stride

Stride

means

```
How many pixels

the filter moves.
```

Example

Stride = 1

```
□□□□□

■■■

↓

Move

1 Pixel
```

Stride = 2

```
Move

2 Pixels
```

Higher Stride

↓

Smaller Output

↓

Less Computation

---

# Padding

Without Padding

Image Size

keeps shrinking.

Example

```
5 × 5

↓

3 × 3

↓

1 × 1
```

Padding

adds

Zeros

around

the image.

Example

```
000000

011110

011110

011110

000000
```

Advantages

✅ Preserve Image Size

✅ Better Edge Detection

---

# Pooling

Pooling

reduces

image size

while preserving

important information.

Benefits

- Faster Training
- Less Memory
- Less Overfitting

---

# Max Pooling

Most common pooling.

Example

```
2 5

7 1

↓

7
```

PyTorch

```python
pool = nn.MaxPool2d(

kernel_size=2,

stride=2
)
```

---

# Average Pooling

Instead of

Maximum

calculate

Average.

```
2 4

6 8

↓

5
```

Used less often than Max Pooling.

---

# CNN Workflow

```
Image

↓

Convolution

↓

ReLU

↓

Pooling

↓

Convolution

↓

ReLU

↓

Pooling

↓

Flatten

↓

Linear Layer

↓

Prediction
```

---

# First CNN in PyTorch

```python
import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(

            in_channels=3,

            out_channels=16,

            kernel_size=3,

            padding=1

        )

        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(

            16,

            32,

            3,

            padding=1

        )

        self.fc = nn.Linear(

            32 * 56 * 56,

            10

        )

    def forward(self,x):

        x = self.pool(

            torch.relu(

                self.conv1(x)
            )
        )

        x = self.pool(

            torch.relu(

                self.conv2(x)
            )
        )

        x = torch.flatten(x,1)

        x = self.fc(x)

        return x
```

---

# Understanding Conv2d

```python
nn.Conv2d(

in_channels=3,

out_channels=32,

kernel_size=3,

stride=1,

padding=1
)
```

Meaning

```
Input

↓

3 Channels

↓

32 Filters

↓

Output

32 Feature Maps
```

---

# Flatten Layer

CNN Output

```
32

×

56

×

56
```

Linear Layer

expects

```
1D Vector
```

Flatten converts

```
3D

↓

1D
```

PyTorch

```python
x = torch.flatten(

x,

1
)
```

---

# Forward Pass

```python
images = torch.randn(

8,

3,

224,

224
)

model = CNN()

outputs = model(images)

print(outputs.shape)
```

Output

```
torch.Size([8,10])
```

Meaning

```
8 Images

↓

10 Classes
```

---

# Training CNN

```python
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(

model.parameters(),

lr=0.001
)

for epoch in range(5):

    for images,labels in train_loader:

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()
```

---

# CNN Architecture Evolution

### LeNet (1998)

- First successful CNN
- Handwritten digit recognition

---

### AlexNet (2012)

- Won ImageNet
- Started Deep Learning revolution

---

### VGG16

- Very deep network
- Simple architecture

---

### GoogLeNet (Inception)

- Parallel convolutions
- More efficient computation

---

### ResNet

Introduced

```
Skip Connections
```

Solved

```
Vanishing Gradient
```

One of the most influential CNN architectures.

---

### EfficientNet

Uses

```
Compound Scaling
```

Balances

- Width
- Depth
- Resolution

---

# CNN vs Fully Connected Network

| Fully Connected | CNN |
|-----------------|-----|
| Huge Parameters | Few Parameters |
| Slow | Faster |
| Ignores Spatial Info | Preserves Spatial Info |
| Overfits Easily | Better Generalization |
| Poor for Images | Excellent for Images |

---

# CNN vs Vision Transformer (ViT)

| CNN | Vision Transformer |
|------|--------------------|
| Convolutions | Self-Attention |
| Local Features | Global Relationships |
| Faster on Small Data | Excels on Large Datasets |
| Inductive Bias | Learns Relationships from Data |

---

# Applications

CNNs are used in

- Face Recognition
- Medical Imaging
- Object Detection
- Self Driving Cars
- OCR
- Satellite Image Analysis
- Defect Detection
- Image Classification

---

# Best Practices

✅ Normalize Images

✅ Use Data Augmentation

✅ Use Batch Normalization

✅ Use Dropout

✅ Start with Pretrained Models

---

# Common Mistakes

❌ Forgetting to normalize images.

❌ Using a very large learning rate.

❌ Ignoring image size changes after pooling.

❌ Feeding incorrectly shaped tensors.

❌ Overfitting due to a small dataset.

---

# Cheat Sheet

| Layer | Purpose |
|---------|----------|
| Conv2d | Feature Extraction |
| ReLU | Non-linearity |
| MaxPool2d | Downsampling |
| Flatten | Convert Feature Maps to Vector |
| Linear | Classification |
| Softmax | Output Probabilities (typically applied during inference; `CrossEntropyLoss` expects raw logits) |

---

# 🤖 How CNNs Work Internally

```
Image

↓

Edges

↓

Corners

↓

Textures

↓

Shapes

↓

Object Parts

↓

Entire Object

↓

Classification
```

Every deeper layer

learns

more complex

features.

---

# Relation with LLMs

CNN

```
Image

↓

Convolution

↓

Prediction
```

Transformer

```
Sentence

↓

Attention

↓

Prediction
```

Both extract features,

but use different mechanisms.

CNNs focus on

local neighborhoods,

while Transformers use

self-attention

to model relationships across the entire input.

---

# Summary

- CNNs are specialized neural networks for image data.
- Convolution extracts local features such as edges and textures.
- Kernels slide over images to produce feature maps.
- Stride controls movement; padding controls output size.
- Pooling reduces computation while preserving important information.
- Modern CNNs build increasingly complex representations through multiple convolutional layers.
- PyTorch provides `nn.Conv2d`, `nn.MaxPool2d`, and related layers to build CNNs efficiently.

---

# 🎤 Interview Questions

1. Why are CNNs better than Fully Connected Networks for images?
2. What is a convolution?
3. What is a kernel?
4. What is the purpose of padding?
5. What is stride?
6. Difference between Max Pooling and Average Pooling?
7. Why do we flatten feature maps?
8. What problem did ResNet solve?
9. Difference between CNN and Vision Transformer?
10. What are feature maps?

---

# 📝 Exercises

### Exercise 1

Build a CNN with:

- 2 Convolution Layers
- 2 MaxPool Layers
- 1 Fully Connected Layer

---

### Exercise 2

Train the CNN on CIFAR-10.

---

### Exercise 3

Experiment with:

- Kernel Size = 3
- Kernel Size = 5

Compare the results.

---

### Exercise 4

Change:

```
Stride

1

↓

2
```

Observe how the output size changes.

---

### Exercise 5

Replace Max Pooling with Average Pooling and compare model performance.

---
