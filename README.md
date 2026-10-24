# Nonholonomic-DDPM

This is a research repository on training nonholonomic DDPMs (Denoising Diffusion Probabilistic Model). Its primary purpose is numerical experimentation for a paper in progress by the author and Matt Jacobs, in which we quantify convergence rates of nonholonomic score-matching to the target distribution.

## What is a nonholonomic DDPM?
The framework for applying DDPMs to solve nonholonomic control problems was first introducced by Elamvazhuthi, Gadginmath, Pasqualetti 2024 (https://arxiv.org/pdf/2402.02297). Here's a summary of the main idea.

The objective is as follows: given a swarm of robots in some initial configuration, can we find a control policy that transports them to a desired target distribution?

It turns out under sufficient 'mixing' properties of the control vectors (known as Hormander's condition), a valid control policy is given by score function of a 'horizontal' diffusion process. This repository trains a neural network to approximate the score function.

Let's consider a simple example. Suppose you have a swarm of unicycle robots. Each robot is constrained in how it can move: it can move forward/backward, and it can steer its direction. How can you move the robots into a desired configuration? 

## Getting started
``` bash
pip install -r requirements.txt
python run.py --g unicycle --target circle --N 5000 --epochs 2000
```

## Methodological details
The methodology is very similar to that of training DDPMs for generative applications, though a key difference is that one cannot use Denoising Score Matching (DSM) to learn the score, since the score of the fundamental solution does not have a clean explicit formula (as in the case of the Ornstein-Uhlenbeck process). Instead we use an IBP formulation of the loss function.

The current repository only supports drift-less systems defined on the torus. We hope to generalize the kinds of systems in the future. 
