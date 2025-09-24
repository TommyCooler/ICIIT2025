# Dataset Structure và Cách Sử Dụng

## Cấu trúc Datasets

Project này hỗ trợ nhiều loại datasets với cấu trúc khác nhau:

### 1. ECG Dataset
```
datasets/ecg/
├── labeled/
│   ├── train/
│   │   ├── chfdb_chf01_275.pkl
│   │   ├── chfdb_chf13_45590.pkl
│   │   └── ... (9 files)
│   └── test/
│       ├── chfdb_chf01_275.pkl
│       ├── chfdb_chf13_45590.pkl
│       └── ... (9 files)
└── raw/
    └── ... (raw text files)
```

### 2. PSM Dataset
```
datasets/psm/
├── train.csv
├── test.csv
└── test_label.csv
```

### 3. NAB Dataset
```
datasets/nab/
├── ambient_temperature_system_failure_train.npy
├── ambient_temperature_system_failure_test.npy
├── ambient_temperature_system_failure_labels.npy
├── ambient_temperature_system_failure.csv
├── labels.json
└── ... (other NAB datasets)
```

### 4. SMAP/MSL Dataset
```
datasets/smap_msl_/
├── train/
│   ├── A-1.npy
│   ├── A-2.npy
│   └── ... (82 files)
├── test/
│   ├── A-1.npy
│   ├── A-2.npy
│   └── ... (82 files)
└── labeled_anomalies.csv
```

### 5. SMD Dataset
```
datasets/smd/
├── machine-1-1.npy
├── machine-1-2.npy
└── ... (84 files)
```

## Cách Sử Dụng

### 1. Kiểm tra cấu trúc datasets
```bash
python check_dataset_structure.py
```

### 2. Test dataset loading
```bash
python simple_dataset_test.py
```

### 3. Test training với ECG
```bash
python test_ecg_training.py
```

### 4. Test toàn bộ pipeline
```bash
python test_complete_pipeline.py
```

### 5. Training và evaluation
```bash
# Train và evaluate
python train_and_evaluate.py --dataset ecg --data_path datasets/ecg --num_epochs 10

# Chỉ evaluate
python test_evaluation.py --dataset ecg --data_path datasets/ecg
```

## Các Scripts Mới

### `check_dataset_structure.py`
- Kiểm tra cấu trúc thư mục datasets
- Xác nhận các file cần thiết có tồn tại
- Hiển thị số lượng files trong mỗi thư mục

### `simple_dataset_test.py`
- Test loading ECG dataset
- Kiểm tra shapes của train/test data
- Xác nhận dataloader hoạt động

### `test_ecg_training.py`
- Test tạo contrastive dataloaders
- Kiểm tra batch loading
- Xác nhận training pipeline

### `test_complete_pipeline.py`
- Test toàn bộ workflow
- Chạy training nhỏ (1 epoch)
- Chạy evaluation
- Hiển thị kết quả tổng hợp

## Các Thay Đổi Trong Code

### 1. ECGDatasetLoader
- Thêm error checking cho file paths
- Hiển thị debug information
- Xử lý lỗi file không tồn tại

### 2. PSMDatasetLoader
- Kiểm tra tồn tại của train.csv, test.csv, test_label.csv
- Thêm debug output

### 3. NABDatasetLoader
- Ưu tiên load pre-processed .npy files
- Fallback về CSV processing nếu cần
- Kiểm tra tồn tại của labels.json

### 4. SMAPMSLDatasetLoader
- Kiểm tra tồn tại của train/ và test/ directories
- Kiểm tra labeled_anomalies.csv

### 5. SMDDatasetLoader
- Kiểm tra tồn tại của .npy files
- Hiển thị data shapes

## Troubleshooting

### Lỗi "No datasets could be loaded"
1. Kiểm tra cấu trúc thư mục: `python check_dataset_structure.py`
2. Xác nhận các file cần thiết có tồn tại
3. Kiểm tra đường dẫn dataset trong command

### Lỗi "File not found"
1. Kiểm tra tên file có đúng không
2. Xác nhận file có tồn tại trong thư mục
3. Kiểm tra quyền đọc file

### Lỗi "Import failed"
1. Đảm bảo đang chạy từ project root directory
2. Kiểm tra conda environment đã được activate
3. Cài đặt dependencies: `pip install -r requirements.txt`

## Ví Dụ Sử Dụng

### Training với ECG dataset
```bash
python src/model/main.py \
    --dataset ecg \
    --data_path datasets/ecg \
    --num_epochs 50 \
    --batch_size 32 \
    --window_size 100
```

### Training với PSM dataset
```bash
python src/model/main.py \
    --dataset psm \
    --data_path datasets/psm \
    --input_dim 25 \
    --num_epochs 50 \
    --batch_size 32
```

### Training với NAB dataset
```bash
python src/model/main.py \
    --dataset nab \
    --data_path datasets/nab \
    --dataset_name ambient_temperature_system_failure \
    --input_dim 1 \
    --num_epochs 50
```

### Training với SMAP/MSL dataset
```bash
python src/model/main.py \
    --dataset smap_msl \
    --data_path datasets/smap_msl_ \
    --dataset_name A-1 \
    --input_dim 25 \
    --num_epochs 50
```

### Training với SMD dataset
```bash
python src/model/main.py \
    --dataset smd \
    --data_path datasets/smd \
    --dataset_name machine-1-1 \
    --input_dim 38 \
    --num_epochs 50
```

## Lưu Ý Quan Trọng

1. **Unsupervised Learning**: Model được train hoàn toàn unsupervised, labels chỉ dùng cho evaluation
2. **Data Preprocessing**: Mỗi dataset có cách xử lý khác nhau
3. **Normalization**: Tự động normalize data nếu `normalize=True`
4. **Validation Split**: Tự động chia test data thành validation và test sets
5. **Error Handling**: Code đã được cải thiện để xử lý lỗi tốt hơn

## Kết Quả

Sau khi chạy thành công, bạn sẽ có:
- Model đã được train
- Checkpoints trong thư mục `checkpoints/`
- Evaluation results trong thư mục `evaluation_results/`
- Plots và visualizations
- Metrics: precision, recall, F1-score, AUC-ROC, AUC-PR
