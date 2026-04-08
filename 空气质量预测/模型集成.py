import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from 数据集 import AirQualityDataset, FEATURE_DIM
from 省份空气质量模型 import AirQualityModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = "weights"

class EnsembleModel:
    def __init__(self, model_paths):
        """初始化集成模型"""
        self.models = []
        
        # 加载所有模型
        for path in model_paths:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
                
                # 从检查点中获取模型配置
                num_provinces = len(checkpoint.get("province_names", [])) or 31
                
                # 创建模型实例
                model = AirQualityModel(
                    num_provinces=num_provinces,
                    seq_len=30,
                    seq_input_dim=FEATURE_DIM,
                ).to(DEVICE)
                
                # 加载模型权重
                model.load_state_dict(checkpoint["model"])
                model.eval()
                self.models.append(model)
                print(f"加载模型: {path}")
            else:
                print(f"模型文件不存在: {path}")
        
        if not self.models:
            raise ValueError("没有加载到任何模型")
        
        print(f"集成模型数量: {len(self.models)}")
    
    def predict(self, province_ids, seq_data):
        """使用集成模型进行预测"""
        with torch.inference_mode():
            predictions = []
            for model in self.models:
                pred = model(province_ids, seq_data)
                predictions.append(pred)
            
            # 计算平均预测
            avg_pred = torch.mean(torch.stack(predictions), dim=0)
            return avg_pred

def evaluate_ensemble(ensemble_model, val_dl):
    """评估集成模型"""
    total_loss = 0
    with torch.inference_mode():
        for pids, inputs, labels in val_dl:
            pids = pids.to(DEVICE)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            pred = ensemble_model.predict(pids, inputs)
            loss = nn.HuberLoss(delta=1.0)(pred, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_dl)
    return avg_loss

def main():
    print("开始模型集成...")
    print(f"设备: {DEVICE}")
    
    # 准备验证数据
    ds_val = AirQualityDataset(seq_len=30, split="val")
    val_dl = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"验证样本数: {len(ds_val)}")
    
    # 收集模型文件
    model_files = []
    for file_name in os.listdir(WEIGHTS_DIR):
        if file_name.endswith(".pt") and file_name != "last_model.pt":
            model_files.append(os.path.join(WEIGHTS_DIR, file_name))
    
    if len(model_files) < 2:
        print("警告: 模型文件不足，至少需要2个模型进行集成")
        return
    
    print(f"找到 {len(model_files)} 个模型文件")
    
    # 创建集成模型
    ensemble_model = EnsembleModel(model_files)
    
    # 评估集成模型
    ensemble_loss = evaluate_ensemble(ensemble_model, val_dl)
    print(f"\n集成模型验证损失: {ensemble_loss:.4f}")
    
    # 与单个模型比较
    print("\n单个模型性能:")
    for path in model_files:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        num_provinces = len(checkpoint.get("province_names", [])) or 31
        
        model = AirQualityModel(
            num_provinces=num_provinces,
            seq_len=30,
            seq_input_dim=FEATURE_DIM,
        ).to(DEVICE)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        
        model_loss = 0
        with torch.inference_mode():
            for pids, inputs, labels in val_dl:
                pids = pids.to(DEVICE)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                pred = model(pids, inputs)
                loss = nn.HuberLoss(delta=1.0)(pred, labels)
                model_loss += loss.item()
        
        model_loss /= len(val_dl)
        print(f"{os.path.basename(path)}: {model_loss:.4f}")
    
    print("\n模型集成完成！")

if __name__ == "__main__":
    main()
