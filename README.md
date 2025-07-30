# uap-detection-deepfake-feature-map

The code in this repository is used for the paper "Robust deepfake detection: Detecting universal adversarial attacks through feature map classification"

- The two surrogate models' weights are provided under `surrogate_models`, which use the MesoInception-4 architecture.

- Four sample feature maps from the validation set extracted from Surrogate Model 2 are provided, one real and one deepfake each for Projected Gradient Descent (PGD) (epsilon 0.09) and Deepfool (epsilon 0.09).

- All best-performing models in the paper can also be found under `models` (Model 3 is located under `max_pool_layer_1_generalisability`)

- All UAP perturbations used in the paper can be found under `UAPs`
