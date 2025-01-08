# CALTECH101 CNN Exercise
Experimenting with different CNN training methods or design choices for classifying Caltech101. This is part of an exercise for a graduate school coursework.

# Default configuration
![hyperparams](https://github.com/user-attachments/assets/f61da338-2df3-49f2-8fe0-9e053076e12b)

# FC layer compression with Truncated SVD
Inspired by Fast R-CNN to compress the FC layer using SVD. Experiment results with different compression ranks/dimensions. Compression seems to decrease the accuracy significantly although model size and parameters are compressed.

![compression](https://github.com/user-attachments/assets/f021304a-9c0a-4e72-abc1-64ed6259c704)

# Impacts of normalization techniques and dropout rate
Custom: 3 Conv blocks (Conv2D, Norm, ReLU, MaxPooling, and DropOut) with 64, 128, and 256 dims each, followed by a 512-dim FC and an output layer.

![custom](https://github.com/user-attachments/assets/f7fe0ae5-80c3-44aa-a0d3-cbdf4ca9b998)

# Training with different initialization
Random init (from scratch) vs ImageNet pre-trained (full optimization) vs ImageNet (FC fine-tune)

![init](https://github.com/user-attachments/assets/0cdec7c0-b52e-4ef5-a490-578b26bcefd3)

# Comparing different loss functions
![loss](https://github.com/user-attachments/assets/60c9a2a6-66ec-44ed-8f26-f7f98925f92e)

# Impacts of other hyperparameters
![etc](https://github.com/user-attachments/assets/0478261b-7279-4b81-97ef-bd89924d52b6)

# Best model (EfficientNet) confusion matrix
Achieving nearly perfect predictions on test set

![conf](https://github.com/user-attachments/assets/aed5f102-d3b0-414b-879a-98958bc7bf90)
