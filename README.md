# OrbitCore
## Dynamic Viewpoint- and Illumination-Invariant Anomaly Detection

> **🛑 STATUS: EXPERIMENTAL PHASE!**
> The algorithm is currently under active testing and fine-tuning; the technology is not yet mature. Comprehensive documentation and a formal publication are in preparation, which will include the precise mathematical and theoretical background, as well as stress-test metrics conducted on industrial datasets (e.g., MVTec AD) as empirical validation, and direct comparative evaluations against the state-of-the-art models.

## 🎯 Project Goal and Concept

Current industrial anomaly detection algorithms (such as PatchCore, PaDiM, etc.) perform perfectly in strictly controlled, fixed-camera production line environments. However, they immediately collapse once the inspected object tilts, the camera viewpoint shifts, or lighting conditions change.

OrbitCore is not another fixed-camera production line algorithm. This engine was designed for dynamic visual environments (e.g., robotic arms, drones, handheld scanners), where perspective and illumination are continuously changing.

## 💡 Novelty and Methodology

* **Test-Time Adaptation Homography in early CNN layer:** The `TTAHomography` module intervenes at runtime via parametric optimization. Instead of rotating raw pixels, it aligns the feature maps (`f_test` and `f_ref`) extracted from the first layer of the network. It iterates the 8 parameters of a $3 \times 3$ projective transformation matrix using the `Adam` optimizer (for a maximum of `n` steps) to maximize cosine similarity (`1 - ECC`) between the test image and the ideal reference view captured during training. The algorithm forces the distorted object back into an upright state before the deeper layers of the neural network and before memory bank search.

* **Feature-Space Orbit Averaging:** The `OrbitCoreModel` breaks with the practice of treating objects as rigid 2D crops. On the feature map, it generates a matrix of 5 different tilt angles (`orbit_params`) based on the specified `orbit_alpha` parameter. The model “tilts” the current view into these directions (`perspective_phis`), then statistically merges them by computing the mean (`orbit_mu`) and standard deviation (`orbit_s`) of the tilted states. The result is a robust feature vector (`z`) that embeds tolerance to micro-movements of the object.

* **Photometric Invariance:** Specular highlights and lighting variations mathematically manifest as an increase in the magnitude of feature vectors. The system eliminates this effect through the ruthless application of `F.normalize(f_srp, p=2, dim=1)`. L2 channel normalization scales each spatial location to unit length (1.0), severely suppressing intensity-derived global illumination variance. For distance computation, primarily the direction of the vector remains, encoding structural and texture information with high tolerance to color and shadow shifts.

* **Dynamic Noise Floor:** The system eliminates static, human-defined thresholds. In the `fit_coreset` method, after building the memory bank, the model tests a random subset of the training data (up to 10,000 samples) against itself (`torch.cdist`). From the resulting distance distribution, it computes the `n%` quantile and designates it as the dynamic noise floor (`self.noise_floor`). During testing, the system subtracts this base noise from the measured distances (via `torch.relu`) and suppresses distortion artifacts occurring at the borders using a dedicated selective masking (`margin = 2`). As a result, only genuine, statistically significant deviations produce anomalies.

* **Automatic Grid Search (Optional):** Built-in hyperparameter optimization engine. When activated, the system autonomously evaluates different Orbit tilt angles (`orbit_alpha`) during training and automatically fixes the parameter that guarantees the maximum decision margin (separability) between the scores of normal and defective samples.

## 🚀 Key Features

* **Zero Defect Data for Deployment:** The core system is purely One-Class based. No prior knowledge or examples of potential defects are required to build the memory bank and deploy the model. (Note: Defective validation samples are only required if the optional Automatic Grid Search is utilized to maximize the decision margin).

* **No Neural Network Training:** No backpropagation and no weight updates are performed. The system uses a pre-trained feature extractor architecture, eliminating the typical hardware and time bottlenecks associated with deep learning.

* **Few-Shot Capability:** Only a few dozen (or however many cover the natural variability of the object) normal, defect-free images are sufficient to build the memory bank. Massive datasets are not required.

* **Dynamic Environmental Robustness:** Unlike rigid baseline models, the engine natively absorbs uncalibrated spatial shifts (perspective tilt, translation) and photometric noise (illumination changes, specular highlights) during inference. This actively eliminates the strict dependency on precision mechanical fixturing and highly controlled lighting environments.

* **Hardware Requirements:** The memory footprint and overall computational load do not currently differ significantly from a standard PatchCore implementation. Precise performance benchmarking and minimum hardware specification tests are still in progress.

### 📦 Example for usage:
```py
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from orbitcore import OrbitCoreLightning

torch.manual_seed(42)

datamodule = MVTecAD(
    root="./datasets/MVTec",
    category="transistor", 
    train_batch_size=16,
    eval_batch_size=4,
)

engine_orbitcore = Engine(accelerator="gpu", devices=1, max_epochs=1)

model_orbitcore = OrbitCoreLightning(
        layers=["layer2", "layer3"], 
        target_dim=128, 
        coreset_sampling_ratio=0.01, 
        use_srp=False, 
        auto_optimize=False,
        orbit_alpha=0.2 
    )

engine_orbitcore.fit(datamodule=datamodule, model=model_orbitcore)

engine_orbitcore.test(datamodule=datamodule, model=model_orbitcore, ckpt_path=None)
```

## 🛡️ License & Citation

This project is open-sourced under the **Apache License 2.0**. It is free to use, modify, and distribute for both academic and commercial purposes, provided that original copyright notices are maintained.

**Academic Use & Citation Policy:**
If you use OrbitCore or its underlying concepts (Test-Time Homography in early CNN layer, Feature-Space Orbit Averaging) in your research, you are expected to cite this repository in your publications.

Please use the "Cite this repository" widget on GitHub, or use the following BibTeX entry:

```bibtex
@software{orbitcore2026,
  author = {Márk Király},
  title = {OrbitCore: Dynamic Viewpoint- and Illumination-Invariant Anomaly Detection},
  year = {2026},
  publisher = {GitHub},
  url = {[https://github.com/ProgrammerGnome/orbitcore](https://github.com/ProgrammerGnome/orbitcore)}
}
