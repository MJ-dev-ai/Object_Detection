# Object Detection with Custom YOLOv5n

This project implements a custom object detection model mimicking YOLOv5n (nano version) using PyTorch. It includes a complete pipeline for training and inference on the VOC dataset.

## Project Structure

```
Object_Detection/
├── config.py              # Configuration file for model and training hyperparameters
├── main.py                # Main entry point for training and validation
├── dataset/
│   ├── __init__.py
│   ├── augmentation.py    # Data augmentation utilities (e.g., Mosaic)
│   └── dataset.py         # VOC dataset loader and preprocessing
├── model/
│   ├── __init__.py
│   ├── common.py          # Common layers (Conv, C3Block, SPPF)
│   └── model.py           # YOLOv5n model architecture (Backbone, Head, Detect)
├── train/
│   ├── __init__.py
│   ├── loss.py            # YOLO loss function (box, obj, cls)
│   └── train.py           # Training loop and optimizer setup
├── utils/
│   ├── __init__.py
│   └── utils.py           # Utility functions (NMS, decoding, visualization)
└── README.md
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- opencv-python
- Pillow
- matplotlib
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Object_Detection
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio tqdm matplotlib opencv-python Pillow numpy
   ```

3. Download VOC2012 dataset and extract to `data/VOC2012` (or update path in `config.py`).

## Usage

### Training

To train the model:
```bash
python main.py
```

Adjust hyperparameters in `config.py` (e.g., batch_size, num_epochs, learning_rate).

### Validation

After training, validation is performed in `main.py` using the `validate` function, which visualizes predictions.

## API Reference

### Model Components

#### `model.model.YOLOv5n(num_classes=20, anchor=3)`
Main model class.

- **Parameters**:
  - `num_classes` (int): Number of object classes (default: 20 for VOC).
  - `anchor` (int): Number of anchors per scale (default: 3).

- **Returns**: Tuple of three tensors for small, medium, large scales.

#### `model.common.Conv(in_channels, out_channels, kernel_size, stride=1, padding=0)`
Convolutional layer with BatchNorm and SiLU activation.

#### `model.common.C3Block(in_channels, out_channels, n=1, shortcut=True)`
CSP bottleneck block.

#### `model.common.SPPF(in_channels, out_channels, k=5)`
Spatial Pyramid Pooling - Fast.

### Loss Function

#### `train.loss.Yolo_Loss(anchors=None, num_classes=20, img_size=640, ...)`
YOLO loss calculator.

- **Parameters**:
  - `anchors` (tensor): Anchor boxes.
  - `num_classes` (int): Number of classes.
  - `img_size` (int): Input image size.

- **Methods**:
  - `forward(preds, targets)`: Computes total loss and components.

### Dataset

#### `dataset.dataset.VOCObjectDetection(root, image_set="train", img_size=640)`
VOC dataset loader.

- **Parameters**:
  - `root` (str): Path to VOC dataset.
  - `image_set` (str): "train", "val", or "trainval".
  - `img_size` (int): Image size.

- **Returns**: Image tensor and target tensor (boxes, labels).

### Utilities

#### `utils.utils.decode_pred(pred, scale_idx, num_classes=20, ...)`
Decodes predictions and applies NMS.

- **Parameters**:
  - `pred` (tensor): Model output for a scale.
  - `scale_idx` (int): Scale index (0: small, 1: medium, 2: large).

- **Returns**: List of detections per batch item.

## Features

- Custom YOLOv5n architecture
- Mosaic augmentation
- VOC dataset support
- Configurable anchors and hyperparameters

## License

This project is licensed under the MIT License.
