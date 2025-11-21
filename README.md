# Image Compression with SVD and FFT

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c.svg)

**A matrix computations project exploring lossy image compression and optimal reconstruction.**

## 📝 Description

This project implements an image compression pipeline that leverages linear algebra and signal processing techniques to reduce image file sizes while maintaining visual fidelity. By treating image channels as matrices, the application applies **Singular Value Decomposition (SVD)** to approximate the image with lower-rank matrices. Additionally, **Fast Fourier Transform (FFT)** is utilized to manipulate image frequencies, allowing for Low-pass (smoothing) and High-pass (sharpening) filtering during the reconstruction process.

The goal is to demonstrate efficient storage methods by decomposing images into their constituent singular values and vectors, storing them in a custom compressed archive, and reconstructing them on demand.

## ✨ Features

* **Channel Separation:** Processes Red, Green, and Blue (RGB) channels independently as matrices.
* **SVD Compression:** Reduces image data redundancy by keeping only the most significant singular values (adjustable compression ratio).
* **Frequency Domain Filtering:** Implements FFT to apply Gaussian Low-pass and High-pass filters for noise reduction or edge enhancement.
* **Custom Artifact Storage:** Saves compressed matrix components (`U`, `S`, `V` matrices) and configuration data into a compact `.zip` format.
* **Reconstruction Pipeline:** Rebuilds images from compressed artifacts with optional post-processing (sharpening/smoothing).

## 📂 File Structure

* **`SAVE.ipynb`**: The compression module. Loads an image, performs SVD/FFT, and saves the compressed components to a zip file.
* **`LOAD.ipynb`**: The reconstruction module. Reads the zip file, reconstructs the matrices, applies filters, and displays the result.
* **`Essentials.py`**: Helper functions for normalization, Gaussian kernel generation, and visualization.
* **`Fourier.py`**: Functions for applying Fourier Transforms and Inverse Fourier Transforms.
* **`Filters.py`**: Implementation of High-pass and Low-pass filter masks.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nikelroid/image-compression.git](https://github.com/nikelroid/image-compression.git)
    cd image-compression
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Install the required libraries using pip:
    ```bash
    pip install numpy opencv-python matplotlib imageio
    ```

## 💻 Usage

### Compressing an Image
1.  Open `SAVE.ipynb` in Jupyter Notebook or VS Code.
2.  Set the target image name in the configuration cell (e.g., `name = 'pepper'`).
3.  Adjust compression parameters like `scale` or `component` thresholds if desired.
4.  Run the notebook. This will generate a `{image_name}.zip` file containing the compressed matrices (U, S, V components) and config.

### Reconstructing an Image
1.  Open `LOAD.ipynb`.
2.  Ensure the generated `.zip` file is in the same directory.
3.  Set the `name` variable to match your compressed file.
4.  Run the notebook to:
    * Extract the matrix components.
    * Reconstruct the RGB channels.
    * Apply Fourier filters (Highpass/Lowpass) for visual enhancement.
    * Display and save the final reconstructed image.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is open-source. [MIT License](LICENSE) (Suggested).

## 📞 Contact

For questions or feedback regarding the Matrix Computations project, please open an issue in the repository.
