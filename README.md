# Generative-Adversarial-Text-to-Image-Synthesis

## 📝 Problem Statement

The objective of this assignment is to implement a **Conditional GAN-based Text-to-Image Synthesis** framework. The model architecture includes three core components:

1. **Source Encoder** – Encodes input images into feature representations.  
2. **Target Generator** – Generates realistic images conditioned on the encoded features and input text descriptions.  
3. **Discriminator** – Distinguishes between real and generated images.

Unlike traditional GANs, the generator here is conditioned not only on random inputs but also on text and learned image representations. The source encoder replaces random noise with more structured inputs, allowing for better control and interpretability.

The system should be trained and tested on a randomly selected subset of 25 classes (20 train + 5 test) from the **Oxford-102 Flowers Dataset**, using textual descriptions from the provided [GitHub repository](https://github.com/taoxugit/AttnGAN/tree/master/data/flowers).

Key requirements:
- All models must be original (no transformers or diffusion models).
- Must be runnable on **Google Colab** and finish within reasonable time (≤ 1 hour preferred).
- Final output includes image grids, t-SNE plots, and model statistics.

The assignment emphasizes model design, training methodology, and creativity in solving the text-to-image synthesis problem.

## My Approach

---

## Source Encoder  
**Purpose**: Encodes an input image into multiple latent vectors and a content feature map.  
**Innovations**:  
- 📌 **Multiple Latents**: Produces *n* distinct latent vectors to enhance diversity and control.  
- 🧩 **Content Representation**: Extracts intermediate spatial features to retain structural content.

---

## Target Generator  
**Purpose**: Generates an image conditioned on both text and visual latent encodings.  
**Innovations**:  
- **Conditional BatchNorm**: Learns adaptive normalization using textual embeddings.  
- **Residual Upsampling**: Employs ResNet-style skip connections for stable upscaling and synthesis.

---

## Discriminator  
**Purpose**: Distinguishes real vs. generated images while aligning with text semantics.  
**Innovations**:  
- **Spectral Normalization**: Stabilizes training by controlling Lipschitz continuity of conv layers.  
- **Text-Image Matching**: Projects text into the discriminator space to enforce semantic alignment.

![image](https://github.com/user-attachments/assets/47776a65-b795-4fa3-ab96-6df9121fb97b)

---

## Training Strategy

**Innovations Introduced:**  
- **Gradient Penalty** in Discriminator for improved stability  
- **Hinge Loss** for both Generator and Discriminator  
- **Contrastive Loss** to align latent space with class labels  
- **Feature Matching Loss** to reduce mode collapse

---

### Training Loop Diagram (Mermaid)

mermaid
flowchart TD
A[Start Epoch] --> B[Sample Batch (Image + Text)]
B --> C[Encode Image to Latents + Content]
C --> D[Generate Fake Image using Text + Latents]

D --> E[Discriminator Forward Pass]
B --> E  %% Real images also go to Discriminator

E --> F[Compute Hinge Loss + Gradient Penalty]
F --> G[Backprop + Update Discriminator]

G --> H[Re-encode + Generate Again]
H --> I[Discriminator on Fake and Real]

I --> J[Compute Generator Loss]
J --> J1[Hinge Loss]
J --> J2[Feature Matching Loss]
J --> J3[Reconstruction Loss]
J --> J4[Contrastive Loss (if class labels present)]

J4 --> K[Total Generator Loss]
K --> L[Backprop + Update Generator + Encoder]

L --> M[Log Epoch Losses]
M --> N{Epoch &#37; 5 == 0?}
N -->|Yes| O[Save Checkpoint]
N -->|No| P[Continue]

O --> Q[Evaluate (Optional)]
P --> Q
Q --> R[Next Epoch or Done]
Loss Functions and Training Objectives

Your training pipeline integrates multiple objectives for stable and semantically aligned generation. Below are the mathematical formulations for each:

1. Hinge Loss

Discriminator Hinge Loss:

$$ \mathcal{L}D ;=; \mathbb{E}{x \sim p_{\text{real}}}\bigl[\max(0,, 1 - D(x))\bigr] ;+; \mathbb{E}{\hat{x} \sim p{\text{fake}}}\bigl[\max(0,, 1 + D(\hat{x}))\bigr] $$

Generator Hinge Loss:

$$ \mathcal{L}G ;=; -,\mathbb{E}{\hat{x} \sim p_{\text{fake}}}\bigl[D(\hat{x})\bigr] $$

2. Gradient Penalty (WGAN-GP-style)

Enforces the Lipschitz constraint by penalizing gradients:

$$ \mathcal{L}{\text{GP}} ;=; \lambda{\text{gp}} ,\cdot, \mathbb{E}{\tilde{x} \sim \mathbb{P}{\text{interp}}} \Bigl[\bigl(|\nabla_{\tilde{x}} D(\tilde{x})|_2 ;-; 1\bigr)^2\Bigr] $$

Where 
x
~
, a linear interpolation of real and fake images.

3. Feature Matching Loss

Matches intermediate discriminator features of real vs. generated images:

$$ \mathcal{L}{\text{FM}} ;=; \sum{l} \bigl|,f_l(x) ;-; f_l(\hat{x})\bigr|_1 $$

Where 
f
l
 are discriminator features at layer 
l
.

4. Contrastive Loss (Optional, Supervised)

Encourages similar latent vectors for same-class images:

$$ \mathcal{L}{\text{contrastive}} ;=; \sum{i < j} \begin{cases} |,z_i - z_j|^2, & \text{if } y_i = y_j, \ \max\bigl(0,,m - |,z_i - z_j|\bigr)^2, & \text{if } y_i \neq y_j, \end{cases} $$

Where 
z
i
 are latent embeddings, and 
m
 is a margin hyperparameter.

5. Reconstruction Loss (L1)

Pixel-level reconstruction loss to align generated and real images:

L
recon

Total Generator Loss

$$ \mathcal{L}{\text{G-total}} ;=; \mathcal{L}G ;+; \lambda{\text{FM}}\cdot \mathcal{L}{\text{FM}} ;+; \lambda_{\text{recon}}\cdot \mathcal{L}{\text{recon}} ;+; \lambda{\text{contrastive}}\cdot \mathcal{L}_{\text{contrastive}} $$

Notes

λ
 values are tunable hyperparameters.
Contrastive loss is used only when class labels are available.
All loss components contribute to a balanced and stable training loop.
Model Parameters

Model	Total Params	Trainable Params	Size (MB)
0 Encoder	22,096,512	22,096,512	84.31
1 Generator	3,534,115	3,534,115	13.52
2 Discriminator	11,675,329	11,675,329	44.63
Results

image image image image image image image

Inception Score: 2.1885813649477743 ± 0.09596595874708262

Quantitative Evaluation Metrics:

inception_score_mean: 2.1885813649477743
inception_score_std: 0.09596595874708262
frechet_inception_distance: 184.25644610545228
References

Miyato, T., et al. (2018). “Spectral Normalization for Generative Adversarial Networks.”
Mirza, M., & Osindero, S. (2014). “Conditional Generative Adversarial Nets.”
Brock, A., Donahue, J., & Simonyan, K. (2019). “Large Scale GAN Training for High Fidelity Natural Image Synthesis.”
Karras, T., et al. (2019). “StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks.”
Karras, T., et al. (2020). “StyleGAN2: Analyzing and Improving the Image Quality of StyleGAN.”
Miyato, T., & Koyama, M. (2018). “ProjGAN: Conditional Generative Adversarial Networks with Projection Discriminator.”
Reed, S., et al. (2016). “Text-to-Image Generation with Deep Learning.”
Tao, M., et al. (2020). “DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis.”
Lim, J. H., & Ye, J. C. (2017). “Geometric GAN.”
Zhang, H., et al. (2019). “Self-Attention Generative Adversarial Networks.”
Mescheder, L., et al. (2018). “Which Training Methods for GANs do actually Converge?”
Gulrajani, I., et al. (2017). “Improved Training of Wasserstein GANs.”
Karras, T., et al. (2018). “Progressive Growing of GANs for Improved Quality, Stability, and Variation.”
