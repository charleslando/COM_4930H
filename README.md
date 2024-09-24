# COM_4930H

#### **Leveraging Jax for Advanced Machine Learning Workflows**

## Table of Contents
1. [Project Proposal](#Project-Proposal)
2. [Learning Jax](#learning-jax)
   1. Why Jax?
   2. Material Used
3. [Reimplementing Existing ML Model With Jax](#reimplementing-existing-ml-model-with-jax)
   1. Previous Model
   2. Training With COCO Dataset
   3. Transformation into Jax
4. [Model Customization and Retraining With Penzai](#model-customization-and-retraining-with-penzai)
5. [Neuron Interpretability](#neuron-interpretability)
6. [Performance Optimization for GPUs and TPUs](#performance-optimizations-for-gpus-and-tpus)
7. [Conclusion](#conclusion)
---

## Project Proposal
Many modern machine learning models demand substantial computational resources and efficient implementation to achieve peak performance. Moreover, retraining these models on new datasets often requires significant time and effort. In this project, I will explore and implement [JAX](https://github.com/jax-ml/jax), a machine learning library that leverages the 'XLA' (Accelerated Linear Algebra) compiler to optimize performance for GPUs and TPUs, providing robust backend and hardware integration.

Building on this knowledge, I will re-create a [previous machine learning model I contributed to](https://github.com/YCCS-Summer-2023-DLCV/Deep-Learning-in-Computer-Vision), initially trained on the [COCO Dataset](https://cocodataset.org/#home), this time using a large database of vehicle images. Additionally, I aim to deepen my understanding of the model’s internal structure by studying the relationships between nodes, focusing on "neuron interpretability" to better explain its decision-making processes.

Lastly, I plan to use the [Penzai](https://github.com/google-deepmind/penzai) framework to make the model adaptable to entirely new datasets without prior training. This will reduce training time, lower computational costs, and enhance the model's adaptability to a wider range of datasets and applications.

---
### Learning Jax
#### Why Jax?
- Optimized for high-performance GPU/TPU usage
- Well-suited for hardware integration
- Functional programming style for concise and readable code
- Easy composition of JAX functions
- Avoids global state due to functional design
- Uses the XLA (Accelerated Linear Algebra) intermediate language
- Seamlessly integrates with backend systems
#### Materials Used
TODO

### Reimplementing Existing ML Model With Jax
Utilize Jax to reimplement an existing deep learning project, including datasets and segmentation tasks. This will demonstrate Jax's capabilities in handling complex machine learning workflows.

#### Previous Model
previous model can be found here
#### Training With Coco Dataset
Using TensorFlow Datasets, I extracted the COCO dataset for my training material, which can be found here
#### Transformation Into Jax
TBD

### Model Customization and Retraining With Penzai
Customize models that need to be retrained for new tasks using Penzai. This will involve adapting existing models to new datasets and tasks, showcasing the flexibility and adaptability of machine learning models built with Jax.

### Neuron Interpretability
Identify which neurons contribute to the classification for each class. This involves exploring techniques in neural network interpretability to understand the decision-making process of the model.

### Performance Optimizations for GPUs and TPUs
TBD

### Conclusion
TBD
