# 🧠 TextSummarizer Using Huggingface (NLP Fine-Tuning Project) 

Course: Complete MLOps Bootcamp With 10+ End-to-End ML Projects
Project Type: End-to-End NLP Project

⚠️ Note: This project is based on a code-along from the course, but has been adapted for a T4 GPU with the following modifications: mixed precision training, reduced gradient accumulation, and gradient checkpointing for efficient training.

## 🚀 Project Overview

An end-to-end NLP project focused on fine-tuning models efficiently while optimizing memory and computation for the available hardware.

## ⚡ Techniques Used

Mixed Precision Training 🔹 – faster computation and lower memory usage

Gradient Checkpointing ♻️ – saves memory during backpropagation

Gradient Accumulation ➕ – simulates larger batch sizes without extra memory

## 💻 Tech Stack

Python

PyTorch / Hugging Face Transformers

### Workflows
1. Config.yaml
2. Params.yaml
3. Config entity
4. Configuration manager
5. Update the components - Data Ingestion, Transformation, Model Trainer
6. Create our pipeline -- Training pipeline, prediction
7. Front end -- API, Training API, Bastch prediction API
