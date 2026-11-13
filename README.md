# Robust Deepfake Detection: Feature Map Classifier with Training Diversity for Universal Adversarial Attacks

This repository contains the code and models for the paper "Robust deepfake detection: Feature map classifier with training diversity for universal adversarial attacks".

This paper is currently still under development and has not been formally published yet.

[Presentation deck](https://docs.google.com/presentation/d/1_E6na9_de6_WLD2rdXjJKMfRWHSx0fZsA-l5FGFdIYM/edit?usp=sharing)

[Manuscript](https://drive.google.com/file/d/1-mra-p8MwureOam9o9jFJPE6YXF492VP/view?usp=sharing)

## Summary

Deepfake detectors are one of the leading methods of combatting deepfake images or photos used in identity fraud or misinformation. However, research has demonstrated that adversarial perturbations added to images are capable of fooling current deepfake models, compromising their effectiveness. While perturbation detectors already exist, a significant challenge lies in ensuring these detectors remain robust and generalisable against unseen perturbation strengths and attack conditions. We hypothesised that increasing the diversity of Universal Adversarial Perturbations (UAPs) within the training dataset would enhance a feature-map-based UAP detector’s generalisability to unseen attacks. Using video frames from the FaceForensics++ dataset, we applied UAPs and extracted feature maps from MesoInception-4. To test our hypothesis, we first developed and optimised a base model, finding that CNNs detecting UAPs on intermediary feature maps in MesoInception-4 generally yielded the best performance. We then compared the performance of our base optimised model trained on limited UAP types against an augmented model with increased UAP diversity in its train set. Our results demonstrated that while the base model exhibited drops in performance when faced with subtle unseen UAPs, the augmented model demonstrated considerably improved accuracy and recall in frame-based and video-based metrics. These findings confirm that training data diversity is crucial for building robust UAP detectors and provides a promising direction for enhancing the overall safety, security and trust in deepfake detection and computer vision against the rising threat of universal adversarial attacks. Future research can further explore broader augmentation strategies and their impact on UAP detectors’ generalisability.

## Repository Contents

- **`requirements.txt`**: Python dependencies.
- **`UAPs/`**: Generated universal adversarial perturbation noises.
- **`samples/`**: Four sample images from two different noises.
- **`models/`**: Pre-trained UAP detector models.
- **`surrogate_models/`**: Surrogate models used to generate UAPs.
- **`scripts/`**:
  - `train.ipynb`: For training models.
  - `test.ipynb`: For testing/evaluating models.
  - `eval.ipynb`: For performance evaluation.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Owen-Tam/uap-detection-deepfake-feature-map.git
    cd uap-detection-deepfake-feature-map
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    ```
    # Optional but recommended: Install Jupyter Lab/Notebook
    pip install jupyterlab
    # or
    # pip install notebook
    ```

## Data Preparation

The notebooks in this repository expect the feature maps extracted from a [MesoInception-4 model](https://github.com/HongguLiu/Deepfake-Detection) predicting on the FaceForensics++ dataset, structured in specific `train` and `test` folders.

1.  **Download FaceForensics++:** Obtain the dataset from the official FaceForensics++ github [FaceForensics++ Github](https://github.com/ondyari/FaceForensics) and by following the instructions on the repository. We used the `raw` and `c23` versions from the DeepFakeDetection Dataset.
2.  **Preprocessing:**

    - Pre-process all videos by extracting frames with faces detected and cropping them to 250x250. You may reference [this script](https://github.com/HongguLiu/Deepfake-Detection/blob/master/detect_from_video.py)
    - Apply the UAPs provided in `UAPs/` to the FaceForensics++ frames.
    - Extract feature maps during inference through pytorch hooks

    ```python
    feature_maps = []
    def get_feature_maps(module, input, output):
        feature_maps.append(output.detach())

    # during inference
    hook_max_pool_1 = model.max_pool_1.register_forward_hook(get_feature_maps)
    torch.save(feature_maps[2][idx], f"./save_path/{frame_id}.pt")

    # after inference
    hook_max_pool_1.remove()
    ```

    - Organize frames into `train` and `test` subdirectories, e.g., `data/train/real/` and `data/train/fake/`.
    - Ensure your data directory structure looks like this:

    ```
    your_ROOT_DIR/
    ├── train/
    │ ├── clean/
    │ └── perturbed/
    └── test/
    │ ├── clean/
    │ └── perturbed/
    ```

## Usage

To run the training, testing, or evaluation:

1.  **Start Jupyter Lab/Notebook:**
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```
2.  **Open the desired notebook** (`train.ipynb`, `test.ipynb`, or `eval.ipynb`) from the `scripts/` directory in your browser.
3.  **Define `ROOT_DIR`:** In `train.ipynb` and `eval.ipynb`, you **must** define `ROOT_DIR` to point to the parent directory containing your prepared `train` and `test` data folders.
    ```python
    # Example: If your data is in /home/user/my_data/train and /home/user/my_data/test
    ROOT_DIR = "/home/user/my_data"
    train_dir = Path(f"{ROOT_DIR}/train")
    test_dir = Path(f"{ROOT_DIR}/test")
    ```
4.  **Run the cells:** Execute the cells in the notebook sequentially.

Check out `./scripts/test.ipynb` for a guide on quickly loading and using one of the pre-trained UAP detector models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or issues, please open an issue on GitHub or contact [owenhwtam@gmail.com](mailto:owenhwtam@gmail.com).
