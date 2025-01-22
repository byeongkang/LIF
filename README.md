# Cross-Domain RGB-to-Event Image Translation Using Spiking Neurons for Eye Detection

This repository provides **Synthetic Event Generation** code for the paper, "**Cross-Domain RGB-to-Event Image Translation Using Spiking Neurons for Eye Detection**." It demonstrates how to transform an Event-style version of the CelebA dataset through several steps involving spiking neuron models, background removal, and region refinement, ultimately generating a synthetic Event dataset ready for eye detection training.

<p align="center">
  <img src="https://user-images.githubusercontent.com/placeholder" width="60%" alt="Example workflow image"/>
</p>


<!--
## Table of Contents
1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [Installation & Requirements](#installation--requirements)
4. [Usage](#usage)
5. [Repository Structure](#repository-structure)
6. [Citation & References](#citation--references)
7. [License](#license)
-->


## Introduction
This repository focuses on generating synthetic Event data from RGB images using **Spiking Neural Networks (SNNs)**. By leveraging the **snntorch** library for the LIF neuron model and an **Adaptive StyleFlow Method** (refer to "*Event Camera-Based Pupil Localization: Facilitating Training With Event-Style Translation of RGB Faces*"), we convert the CelebA dataset into an Event-style dataset. The process includes:

- **Event-Style Conversion** of CelebA images
- **Background Removal** using face segmentation and parsing
- **Region-Aware Polarity Generation & Refinement** for key facial regions

The final output is a **Synthetic-Event Dataset** suitable for training your **Detection Network** on tasks such as eye or pupil detection.

---

## Pipeline Overview
Below is a high-level outline of the steps involved:

1. **Event-Style Dataset Creation**
   - Convert the **CelebA dataset** ([Official Link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) into an Event-like domain via the **Adaptive StyleFlow Method**.
   - Please see the paper "*Event Camera-Based Pupil Localization: Facilitating Training With Event-Style Translation of RGB Faces*" for the core method.

2. **LIF Neuron Processing** (`LIF_generator.py`)
   - Utilize [snntorch](https://github.com/jeshraghian/snntorch) to implement the **Leaky Integrate-and-Fire (LIF)** neuron model.
   - Load the Event-style dataset and feed it through the LIF neuron model to generate spiking activities mimicking event data.

3. **Face Cropping** (Using [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch))
   - Obtain segmentation and boundary information of facial regions for accurate cropping.

4. **Background Removal** (`remove_background.py`)
   - Removes the background of the cropped CelebA images based on facial segmentation data.

5. **Region-Aware Polarity Generation & Refinement** (`region_refinement.py`)
   - Generates **Event Polarity** for facial keypoints.
   - Refines the **eye region** and other important facial areas to produce highly accurate Event representations.

6. **Final Synthetic-Event Dataset**
   - The resulting dataset can be used to **train your Detection Network**, particularly for **pupil or eye detection** in an Event-based setting.

---

