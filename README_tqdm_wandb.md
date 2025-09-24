# Tqdm và Wandb Integration

## 📊 **Tổng quan**

Dự án đã được tích hợp với **tqdm** để theo dõi progress bar và **wandb** để logging metrics trong quá trình training.

## 🚀 **Cài đặt**

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup wandb
```bash
# Cách 1: Login qua CLI
wandb login

# Cách 2: Set API key
export WANDB_API_KEY=your_api_key_here

# Cách 3: Chạy setup script
python setup_wandb.py
```

## 📈 **Tqdm Progress Bars**

### **Training Progress**
```
Training: 100%|██████████| 50/50 [02:30<00:00, 3.00s/it, Loss=0.1234, Contrastive=0.0567, Reconstruction=0.0667]
```

### **Validation Progress**
```
Validation: 100%|██████████| 10/10 [00:30<00:00, 3.00s/it, Loss=0.1456, Contrastive=0.0678, Reconstruction=0.0778]
```

### **Features**
- ✅ Real-time loss display
- ✅ Progress percentage
- ✅ Time estimation
- ✅ Speed (iterations/second)
- ✅ Color-coded progress bars

## 📊 **Wandb Logging**

### **Batch-level Metrics**
- `batch/total_loss`: Total loss per batch
- `batch/contrastive_loss`: Contrastive loss per batch
- `batch/reconstruction_loss`: Reconstruction loss per batch
- `batch/learning_rate`: Learning rate per batch

### **Epoch-level Metrics**
- `epoch/train_total_loss`: Average training loss per epoch
- `epoch/train_contrastive_loss`: Average contrastive loss per epoch
- `epoch/train_reconstruction_loss`: Average reconstruction loss per epoch
- `epoch/val_total_loss`: Average validation loss per epoch
- `epoch/val_contrastive_loss`: Average validation contrastive loss per epoch
- `epoch/val_reconstruction_loss`: Average validation reconstruction loss per epoch
- `epoch/learning_rate`: Learning rate per epoch
- `epoch/epoch`: Epoch number

### **Validation Metrics**
- `val_batch/total_loss`: Validation loss per batch
- `val_batch/contrastive_loss`: Validation contrastive loss per batch
- `val_batch/reconstruction_loss`: Validation reconstruction loss per batch

## 🎯 **Sử dụng**

### **1. Training với Wandb (Mặc định)**
```bash
python src/model/main.py --dataset ecg --epochs 10 --use_wandb
```

### **2. Training không Wandb**
```bash
python src/model/main.py --dataset ecg --epochs 10 --no_wandb
```

### **3. Custom Wandb Settings**
```bash
python src/model/main.py \
    --dataset ecg \
    --epochs 10 \
    --use_wandb \
    --project_name "my-project" \
    --experiment_name "experiment-1"
```

### **4. Demo Scripts**
```bash
# Demo đầy đủ với wandb
python demo_tqdm_wandb.py

# Demo đơn giản
python simple_wandb_demo.py

# Setup wandb
python setup_wandb.py
```

## 🔧 **Configuration**

### **Wandb Arguments**
```python
# Trong main.py
parser.add_argument('--use_wandb', action='store_true', default=True)
parser.add_argument('--no_wandb', dest='use_wandb', action='store_false')
parser.add_argument('--project_name', type=str, default='contrastive-learning')
parser.add_argument('--experiment_name', type=str, default=None)
```

### **Trainer Arguments**
```python
# Trong ContrastiveTrainer
trainer = ContrastiveTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    use_wandb=True,  # Enable/disable wandb
    project_name='contrastive-learning',
    experiment_name='my-experiment'
)
```

## 📱 **Wandb Dashboard**

### **Truy cập Dashboard**
1. Đi tới [https://wandb.ai](https://wandb.ai)
2. Login vào tài khoản
3. Navigate đến project `contrastive-learning`
4. Xem run `my-experiment`

### **Metrics Visualization**
- **Loss Curves**: Training và validation loss theo thời gian
- **Learning Rate**: Learning rate schedule
- **Batch Metrics**: Real-time metrics per batch
- **System Metrics**: CPU, GPU usage (nếu có)

## 🎨 **Customization**

### **Thêm Custom Metrics**
```python
# Trong train_epoch()
if self.use_wandb:
    wandb.log({
        'custom/my_metric': custom_value,
        'custom/another_metric': another_value
    })
```

### **Custom Progress Bar**
```python
# Trong train_epoch()
pbar = tqdm(self.train_dataloader, desc="Training", 
           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
```

### **Custom Wandb Config**
```python
# Trong __init__
wandb.init(
    project=project_name,
    name=experiment_name,
    config={
        'custom_param': custom_value,
        'model_architecture': 'TCN+Transformer',
        'dataset': 'ECG'
    }
)
```

## 🐛 **Troubleshooting**

### **Wandb Issues**
```bash
# Check wandb status
wandb status

# Login again
wandb login

# Check API key
echo $WANDB_API_KEY
```

### **Tqdm Issues**
```python
# Disable tqdm
from tqdm import tqdm
tqdm.disable = True

# Or use tqdm with different settings
pbar = tqdm(dataloader, disable=True)
```

### **Common Errors**
1. **Wandb not logged in**: Run `wandb login`
2. **API key invalid**: Check `WANDB_API_KEY` environment variable
3. **Project not found**: Create project on wandb dashboard
4. **Tqdm not showing**: Check if running in terminal (not IDE)

## 📚 **Examples**

### **Basic Training**
```python
from src.model.train_contrastive import ContrastiveTrainer
from src.model.contrastive_model import ContrastiveModel

# Create model
model = ContrastiveModel(input_dim=2, d_model=64)

# Create trainer with wandb
trainer = ContrastiveTrainer(
    model=model,
    train_dataloader=train_dataloader,
    use_wandb=True,
    project_name='my-project'
)

# Train
history = trainer.train(num_epochs=10)
```

### **Custom Logging**
```python
# Log custom metrics
wandb.log({
    'custom/accuracy': accuracy,
    'custom/f1_score': f1_score,
    'custom/precision': precision,
    'custom/recall': recall
})
```

## 🎯 **Best Practices**

1. **Wandb**:
   - Sử dụng tên experiment có ý nghĩa
   - Group các runs liên quan
   - Log hyperparameters quan trọng
   - Sử dụng tags để organize

2. **Tqdm**:
   - Sử dụng desc để mô tả rõ ràng
   - Customize postfix để hiển thị metrics quan trọng
   - Sử dụng unit để hiển thị đơn vị

3. **Logging**:
   - Log cả batch-level và epoch-level metrics
   - Sử dụng prefix để group metrics
   - Log learning rate và other hyperparameters

## 🔗 **Links**

- [Wandb Documentation](https://docs.wandb.ai/)
- [Tqdm Documentation](https://tqdm.github.io/)
- [Wandb Dashboard](https://wandb.ai)
- [Project Repository](#)

---

**Happy Training! 🚀**
