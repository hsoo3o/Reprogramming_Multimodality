# Reprogramming Multimodality

The development of multimodal deep learning models has attracted significant interest due to their ability to integrate diverse data sources. A common approach is to learn a shared multi-modal embedding space, but this often requires separate encoders for each modality, increasing computational costs.

To address this, we propose a reprogramming technique that maps inputs from various domains (e.g., time-series data) into a common image domain using an attention-based reprogramming layer. While this approach reduces computational overhead, experiments reveal that its performance remains limited compared to existing methods, highlighting challenges in achieving effective multimodal representation learning through reprogramming.





## Usage

1. Clone the Repository


2. Install Dependencies

    ```bash
    pip install -r requirements.txt
    ```


3. Prepare Dataset

    We used the following datasets for our experiments:
    - [Sun RGB-D](https://rgbd.cs.princeton.edu/)
    - [NYU-D](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
    - [Audioset](https://research.google.com/audioset/)
    - [ESC-50](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file)


    To set up the datasets:

    1. Create a dataset directory:
        ```bash
        mkdir ./dataset
        ```

    2. Download and place the datasets into the `./dataset` directory.


4. Get pretrained weight

    - For training from scratch: [reprogram_base](https://drive.google.com/file/d/1E5vQFz-gtzZyn--Hh9EogojwOcFfP2CG/view?usp=sharing)
    - For testing pretrained weight: [reprogram_trained](https://drive.google.com/file/d/1Nc35W9sER1sTCXFLMDFh7tFs54J_5Y6W/view?usp=sharing)

5. Train and Test the Reprogramming Layer

    - For **audio reprogramming**, use the `audio_train_n_test.ipynb` notebook.
    - For **depth reprogramming**, use the `depth_train_n_test.ipynb` notebook.


## Acknowledgements
This project was heavily inspired by the [ImageBind](https://github.com/facebookresearch/ImageBind) repository and [OpenCLIP](https://github.com/mlfoundations/open_clip) repository and leverages some of their core principles.
