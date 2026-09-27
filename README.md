# Conditional GAN Image Generation

A deep learning project that uses a **Conditional Generative Adversarial Network (CGAN)** to generate grayscale images conditioned on class labels.

The project uses **Car** and **Plane** images and compares a baseline CGAN with an improved architecture designed to stabilize adversarial training.

## Problem

A Conditional GAN extends a standard GAN by providing class information to both the Generator and Discriminator. This allows image generation to be conditioned on a selected class.

Target classes:

- Car
- Plane

## Architecture

### Generator

The Generator receives random noise and a class-label embedding and produces a synthetic image.

### Discriminator

The Discriminator receives an image together with its class-label embedding and predicts whether the image is real or generated.

```text
Noise + Class Label
        ↓
    Generator
        ↓
 Generated Image
        ↓
Image + Class Label
        ↓
  Discriminator
        ↓
    Real / Fake
```

## Baseline CGAN

The baseline follows the assignment architecture using fully connected layers.

**Generator:**

```text
Noise + Label Embedding
→ 128
→ 256
→ 512
→ 1024
→ Image
```

**Discriminator:**

```text
Image + Label Embedding
→ 512
→ 1024
→ 1024
→ 512
→ Real / Fake
```

## Improved CGAN

The improved architecture introduces:

- Deeper Generator
- Batch Normalization in the Generator
- Spectral Normalization in the Discriminator
- Dropout in the Discriminator
- One-sided label smoothing
- Smaller learning rate
- Larger noise dimension
- 200 training epochs

These changes were intended to reduce Discriminator dominance and make adversarial training more stable.

## Results

The models are evaluated using **Fréchet Inception Distance (FID)**, where a lower value indicates a closer distribution between generated and real images.

| Model | FID |
|---|---:|
| Baseline CGAN | 125.05 |
| Improved CGAN | **90.22** |

The Improved CGAN reduced FID by **34.83 points**.

The training losses also became more stable, with the Discriminator loss remaining closer to `ln(2) ≈ 0.693`, indicating better balance between the Generator and Discriminator.

However, the generated images were still far from realistic, and the FID estimate has limitations because only 200 samples were used and the covariance calculation produced a singular-matrix warning.

## Key Takeaway

The experiment shows that stabilizing a CGAN with Batch Normalization, Spectral Normalization, Dropout, label smoothing, and adjusted training settings can improve training balance and FID.

However, the fully connected architecture remains limited for image generation. A convolutional architecture such as DCGAN would be a natural direction for future improvement.
