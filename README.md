# eTensor

Let’s Build a Deep Learning Library from Scratch Using NumPy (Part 1)

## Introduction

We are building a PyTorch-like deep learning library from scratch called **eTensor**.
Starting from a blank Python file and NumPy, the goal is to keep building until we have:

- A functional autograd engine
- Trainable models
- Practical examples like MNIST and CNNs

## What This Series Is About

This is **not** a tutorial on using existing deep learning libraries.

Instead, we will:

- Start from a blank Python file
- Wrap NumPy arrays in a Tensor abstraction
- Track tensor operations
- Build a computation graph
- Implement backpropagation ourselves

## Current Scope (Part 1)

In this part, we begin with the core tensor and operation foundations that will power automatic differentiation in future parts.
