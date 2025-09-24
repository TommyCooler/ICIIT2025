# Contrastive Learning for Time Series Anomaly Detection

Model contrastive learning sử dụng TCN + Transformer encoder để học representation từ time series data với data augmentation.

## Kiến trúc Model

### 1. Encoder (TCN + Transformer)
- **TCN Block**: Temporal Convolutional Network với dilated convolutions
- **Transformer Block**: Multi-head attention với positional encoding
- **Combination**: Concatenate hoặc stack outputs từ TCN và Transformer

### 2. Contrastive Learning Branch
- **MLP Projection**: Project encoded features xuống dimension thấp hơn
- **InfoNCE Loss**: Contrastive loss giữa original và augmented data
- **Positive pairs**: Original data + augmentation của chính nó
- **Negative pairs**: Original data + augmentation của data khác

### 3. Reconstruction Branch
- **Decoder**: TCN + Transformer decoder để reconstruct original data
- **Reconstruction Loss**: MSE loss giữa original và reconstructed data

## Cấu trúc Dữ liệu

### Window Sampling
- **Non-overlapping windows**: Mỗi window không chồng lên nhau
- **Random sampling**: Mỗi batch được sample ngẫu nhiên từ tập windows
- **Data augmentation**: Áp dụng augmentation với xác suất nhất định

### Data Augmentation
1. **Add Noise**: Thêm Gaussian noise
2. **Time Warp**: Biến dạng thời gian
3. **Magnitude Warp**: Biến dạng biên độ
4. **Window Slice**: Cắt ngẫu nhiên một phần window
5. **Window Warp**: Biến dạng một phần window

## Cách sử dụng

### 1. Training từ command line

```bash
# Training với ECG dataset
python src/model/main.py --dataset_type ecg --data_path datasets/ecg --num_epochs 100

# Training với PSM dataset
python src/model/main.py --dataset_type psm --data_path datasets/psm --input_dim 25 --num_epochs 100

# Training với custom parameters
python src/model/main.py \
    --dataset_type ecg \
    --data_path datasets/ecg \
    --window_size 100 \
    --batch_size 32 \
    --num_epochs 100 \
    --d_model 256 \
    --projection_dim 128 \
    --learning_rate 1e-4 \
    --contrastive_weight 1.0 \
    --reconstruction_weight 1.0
```

### 2. Sử dụng trong code

```python
from src.model.contrastive_model import ContrastiveModel, ContrastiveDataset
from src.model.train_contrastive import ContrastiveTrainer, create_contrastive_dataloaders

# Tạo dataloaders
train_dataloader, val_dataloader = create_contrastive_dataloaders(
    dataset_type='ecg',
    data_path='datasets/ecg',
    window_size=100,
    batch_size=32,
    augmentation_prob=0.5
)

# Tạo model
model = ContrastiveModel(
    input_dim=2,  # ECG có 2 channels
    d_model=256,
    projection_dim=128,
    nhead=8,
    transformer_layers=6,
    dropout=0.1
)

# Tạo trainer
trainer = ContrastiveTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    learning_rate=1e-4,
    contrastive_weight=1.0,
    reconstruction_weight=1.0
)

# Training
history = trainer.train(num_epochs=100)
```

### 3. Demo script

```bash
# Chạy demo
python demo_contrastive.py
```

## Tham số Model

### Model Parameters
- `input_dim`: Số features đầu vào (ECG: 2, PSM: 25, NAB: 1, SMAP/MSL: 25, SMD: 38)
- `d_model`: Dimension của model (mặc định: 256)
- `projection_dim`: Dimension cho contrastive learning (mặc định: 128)
- `nhead`: Số attention heads (mặc định: 8)
- `transformer_layers`: Số layers của transformer (mặc định: 6)
- `tcn_output_dim`: Output dimension của TCN (mặc định: None)
- `tcn_kernel_size`: Kernel size của TCN (mặc định: 3)
- `tcn_num_layers`: Số layers của TCN (mặc định: 3)
- `dropout`: Dropout rate (mặc định: 0.1)
- `temperature`: Temperature cho InfoNCE loss (mặc định: 0.07)
- `combination_method`: Cách kết hợp TCN và Transformer ('concat' hoặc 'stack')

### Training Parameters
- `window_size`: Kích thước window (mặc định: 100)
- `batch_size`: Batch size (mặc định: 32)
- `num_epochs`: Số epochs (mặc định: 100)
- `learning_rate`: Learning rate (mặc định: 1e-4)
- `weight_decay`: Weight decay (mặc định: 1e-5)
- `contrastive_weight`: Trọng số cho contrastive loss (mặc định: 1.0)
- `reconstruction_weight`: Trọng số cho reconstruction loss (mặc định: 1.0)
- `augmentation_prob`: Xác suất áp dụng augmentation (mặc định: 0.5)

## Loss Functions

### 1. Contrastive Loss (InfoNCE)
```python
# Positive pairs: original + augmentation của chính nó
# Negative pairs: original + augmentation của data khác
contrastive_loss = F.cross_entropy(similarity_matrix, labels)
```

### 2. Reconstruction Loss (MSE)
```python
# Reconstruct original data từ augmented encoding
reconstruction_loss = F.mse_loss(reconstructed, original)
```

### 3. Total Loss
```python
total_loss = contrastive_weight * contrastive_loss + reconstruction_weight * reconstruction_loss
```

## Cấu trúc File

```
src/
├── model/
│   ├── main.py                 # Main training script
│   ├── contrastive_model.py    # Contrastive learning model
│   └── train_contrastive.py    # Training utilities
├── modules/
│   ├── encoder.py              # TCN + Transformer encoder
│   ├── decoder.py              # TCN + Transformer decoder
│   └── augmentation.py         # Data augmentation modules
└── utils/
    └── dataloader.py           # Data loading utilities

demo_contrastive.py             # Demo script
README_contrastive.md           # Documentation
```

## Ví dụ Training

### ECG Dataset
```bash
python src/model/main.py \
    --dataset_type ecg \
    --data_path datasets/ecg \
    --window_size 100 \
    --batch_size 32 \
    --num_epochs 100 \
    --d_model 256 \
    --projection_dim 128 \
    --learning_rate 1e-4
```

### PSM Dataset
```bash
python src/model/main.py \
    --dataset_type psm \
    --data_path datasets/psm \
    --input_dim 25 \
    --window_size 50 \
    --batch_size 64 \
    --num_epochs 100
```

### NAB Dataset
```bash
python src/model/main.py \
    --dataset_type nab \
    --data_path datasets/nab \
    --dataset_name ambient_temperature_system_failure \
    --input_dim 1 \
    --window_size 100 \
    --batch_size 32
```

## Monitoring Training

### Checkpoints
- Model được lưu mỗi `save_every` epochs
- Best model được lưu dựa trên validation loss
- Configuration được lưu trong `config.json`

### Training History
- Plot training/validation losses
- Plot contrastive và reconstruction losses
- Plot learning rate schedule

### Logging
- Progress bar với real-time metrics
- Epoch summary với losses
- Model parameters count

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Giảm `batch_size`
   - Giảm `window_size`
   - Giảm `d_model` hoặc `projection_dim`

2. **Slow Training**
   - Tăng `num_workers`
   - Sử dụng GPU nếu có
   - Giảm `transformer_layers` hoặc `tcn_num_layers`

3. **Poor Performance**
   - Tăng `num_epochs`
   - Điều chỉnh `learning_rate`
   - Thay đổi `contrastive_weight` và `reconstruction_weight`
   - Thử các `combination_method` khác nhau

### Debug Mode
```bash
python src/model/main.py --verbose --device cpu
```

## Dependencies

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib tqdm
```

## Citation

Nếu bạn sử dụng code này trong nghiên cứu, vui lòng cite:

```bibtex
@article{contrastive_timeseries,
  title={Contrastive Learning for Time Series Anomaly Detection},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```
