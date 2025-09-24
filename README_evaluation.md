# Evaluation Module for Contrastive Learning Model

Module này cung cấp các công cụ để đánh giá hiệu suất của model contrastive learning trong việc phát hiện anomaly.

## Tính năng chính

### 1. AnomalyEvaluator Class
- **Tính toán reconstruction error**: Đo lường mức độ model có thể reconstruct lại dữ liệu gốc
- **Tìm threshold tối ưu**: Tự động tìm ngưỡng phát hiện anomaly dựa trên F1-score, precision, recall, hoặc AUC
- **Đánh giá hiệu suất**: Tính toán các metrics như precision, recall, F1-score, AUC-ROC, AUC-PR
- **Visualization**: Tạo các biểu đồ phân tích anomaly scores

### 2. Evaluation Scripts
- **test_evaluation.py**: Script chính để đánh giá model đã được train
- **train_and_evaluate.py**: Script để train model và đánh giá liên tiếp
- **quick_evaluation_test.py**: Script test nhanh các tính năng evaluation

## Cách sử dụng

### 1. Đánh giá model đã được train

```bash
# Sử dụng checkpoint mới nhất
python test_evaluation.py --dataset ecg --data_path datasets/ecg

# Sử dụng checkpoint cụ thể
python test_evaluation.py --dataset ecg --data_path datasets/ecg --model_path checkpoints/ecg_20250924_100017/final_model.pt

# Tùy chỉnh parameters
python test_evaluation.py \
    --dataset ecg \
    --data_path datasets/ecg \
    --window_size 100 \
    --batch_size 32 \
    --threshold_method f1 \
    --save_plots \
    --plot_anomalies
```

### 2. Train và đánh giá liên tiếp

```bash
# Train 10 epochs và đánh giá
python train_and_evaluate.py --dataset ecg --data_path datasets/ecg --num_epochs 10

# Chỉ đánh giá (bỏ qua training)
python train_and_evaluate.py --dataset ecg --data_path datasets/ecg --skip_training

# Tùy chỉnh training parameters
python train_and_evaluate.py \
    --dataset ecg \
    --data_path datasets/ecg \
    --num_epochs 20 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --d_model 512
```

### 3. Test nhanh evaluation module

```bash
python quick_evaluation_test.py
```

## Các tham số quan trọng

### Dataset Parameters
- `--dataset`: Loại dataset (ecg, psm, nab, smap_msl, smd)
- `--data_path`: Đường dẫn đến dataset
- `--dataset_name`: Tên dataset cụ thể (cho nab, smap_msl, smd)

### Model Parameters
- `--input_dim`: Số chiều input (số features)
- `--d_model`: Kích thước model dimension
- `--projection_dim`: Kích thước projection dimension
- `--window_size`: Kích thước window
- `--batch_size`: Batch size

### Evaluation Parameters
- `--threshold_method`: Phương pháp tìm threshold (f1, precision, recall, auc)
- `--custom_threshold`: Threshold tùy chỉnh (nếu không muốn tự động tìm)
- `--save_plots`: Lưu các biểu đồ phân tích
- `--plot_anomalies`: Vẽ kết quả phát hiện anomaly

## Kết quả đánh giá

### Metrics được tính toán
- **Precision**: Tỷ lệ dự đoán đúng trong số các dự đoán anomaly
- **Recall**: Tỷ lệ phát hiện được các anomaly thực tế
- **F1-Score**: Harmonic mean của precision và recall
- **AUC-ROC**: Area Under ROC Curve
- **AUC-PR**: Area Under Precision-Recall Curve
- **Anomaly Ratio**: Tỷ lệ dữ liệu được phát hiện là anomaly

### Output files
- `evaluation_results.txt`: Kết quả đánh giá dạng text
- `anomaly_scores.png`: Biểu đồ phân tích anomaly scores
- `anomaly_detection.npz`: Kết quả phát hiện anomaly (scores, predictions, indices)

## Ví dụ kết quả

```
============================================================
EVALUATION RESULTS
============================================================
Threshold: 0.123456
Precision: 0.8500
Recall: 0.7800
F1-Score: 0.8136
AUC-ROC: 0.9200
AUC-PR: 0.8800
Anomaly Ratio: 0.0500
Mean Score: 0.098765
Std Score: 0.012345
============================================================
```

## Lưu ý quan trọng

### 1. Labels chỉ dùng cho evaluation
- Model được train hoàn toàn unsupervised (không sử dụng labels)
- Labels chỉ được sử dụng để đánh giá hiệu suất sau khi training
- Điều này đảm bảo tính hợp lệ của unsupervised learning approach

### 2. Threshold selection
- Mặc định sử dụng F1-score để tìm threshold tối ưu
- Có thể tùy chỉnh threshold hoặc phương pháp tìm threshold
- Threshold được tính dựa trên reconstruction error

### 3. Dataset compatibility
- Hỗ trợ tất cả các dataset types: ecg, psm, nab, smap_msl, smd
- Tự động detect input dimension từ dataset
- Xử lý labels khác nhau cho từng loại dataset

## Troubleshooting

### Lỗi thường gặp

1. **"No trained model found"**
   - Đảm bảo đã train model trước khi đánh giá
   - Kiểm tra đường dẫn checkpoint directory

2. **"Could not create evaluation dataloader"**
   - Kiểm tra đường dẫn dataset
   - Đảm bảo dataset có đầy đủ train/test data

3. **"No anomalies in labels"**
   - Một số dataset có thể không có labels
   - Sử dụng percentile-based threshold (top 5% scores)

### Performance tips
- Sử dụng GPU nếu có sẵn (`--device cuda`)
- Tăng `--num_workers` để tăng tốc data loading
- Giảm `--batch_size` nếu gặp lỗi memory

## Tích hợp với existing code

Evaluation module được thiết kế để tích hợp seamlessly với existing codebase:

```python
from src.utils.evaluation import AnomalyEvaluator, create_evaluation_dataloader

# Load trained model
model = ContrastiveModel(...)
model.load_state_dict(torch.load('checkpoint.pt'))

# Create evaluator
evaluator = AnomalyEvaluator(model)

# Create evaluation dataloader
eval_dataloader = create_evaluation_dataloader('ecg', 'datasets/ecg')

# Run evaluation
metrics = evaluator.evaluate(eval_dataloader)
print(f"F1-Score: {metrics['f1_score']:.4f}")
```
