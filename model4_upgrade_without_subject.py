import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def augment_data(titles, summaries, ratings):
    augmented_titles = []
    augmented_summaries = []
    augmented_ratings = []
    
    for title, summary, rating in zip(titles, summaries, ratings):
        # 处理可能的 NaN 值
        title = str(title) if not pd.isna(title) else ""
        summary = str(summary) if not pd.isna(summary) else ""
        
        # 原始数据
        augmented_titles.append(title)
        augmented_summaries.append(summary)
        augmented_ratings.append(rating)
        
        # 添加噪声到评分
        noisy_rating = rating + np.random.normal(0, 0.1)
        augmented_titles.append(title)
        augmented_summaries.append(summary)
        augmented_ratings.append(noisy_rating)
        
        # 截断或扩展摘要
        if len(summary) > 50:
            short_summary = summary[:50]
            augmented_titles.append(title)
            augmented_summaries.append(short_summary)
            augmented_ratings.append(rating)
    
    return augmented_titles, augmented_summaries, augmented_ratings

class AnimeDataset(Dataset):
    def __init__(self, titles, summaries, ratings, tokenizer, max_length=512):
        self.titles = titles
        self.summaries = summaries
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = str(self.titles[idx]) if not pd.isna(self.titles[idx]) else ""
        summary = str(self.summaries[idx]) if not pd.isna(self.summaries[idx]) else ""
        rating = float(self.ratings[idx])

        if not title and not summary:
            text = "无内容"
        else:
            text = f"{title} {summary}".strip()

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(rating, dtype=torch.float)
        }

class AnimeRatingModel(nn.Module):
    def __init__(self, bert_model, dropout_rate=0.3):
        super(AnimeRatingModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

def train(model, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ratings = batch['rating'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), ratings)
            total_loss += loss.item()

            predictions = outputs.squeeze()
            correct_predictions += torch.sum(torch.abs(predictions - ratings) <= 1).item()
            total_predictions += ratings.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    correct_predictions_05 = 0
    total_predictions = 0
    all_predictions = []
    all_true_ratings = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), ratings)
            total_loss += loss.item()

            predictions = outputs.squeeze()
            correct_predictions += torch.sum(torch.abs(predictions - ratings) <= 1).item()
            correct_predictions_05 += torch.sum(torch.abs(predictions - ratings) <= 0.5).item()
            total_predictions += ratings.size(0)

            all_predictions.extend(predictions.cpu().tolist())
            all_true_ratings.extend(ratings.cpu().tolist())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    accuracy_05 = correct_predictions_05 / total_predictions
    
    print(f'\nTest Loss: {avg_loss:.4f}')
    print(f'Test Accuracy (predictions within 1 point): {accuracy:.4f}')
    print(f'Test Accuracy (predictions within 0.5 points): {accuracy_05:.4f}')
    
    return avg_loss, accuracy, accuracy_05, all_predictions, all_true_ratings

def main():
    # 尝试不同的编码方式读取CSV文件
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'iso-8859-1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv('anime_data.csv', encoding=encoding)
            print(f"Successfully read the file with {encoding} encoding.")
            break
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding. Trying next...")
    
    if df is None:
        print("Failed to read the file with all attempted encodings.")
        return

    # 检查并打印数据类型
    print("\nColumn dtypes:")
    print(df.dtypes)
    
    # 检查并打印是否有缺失值
    print("\nMissing values:")
    print(df.isnull().sum())

    # 检查并打印一些样本数据
    print("\nSample data:")
    print(df.head())

    titles = df['title'].tolist()
    summaries = df['subject_summary'].tolist()
    ratings = df['rating'].tolist()

    # 应用数据增强
    titles, summaries, ratings = augment_data(titles, summaries, ratings)

    # 首先分割出测试集
    train_val_titles, test_titles, train_val_summaries, test_summaries, train_val_ratings, test_ratings = train_test_split(
        titles, summaries, ratings, test_size=0.2, random_state=42)

    # 然后将剩余的数据分割为训练集和验证集
    train_titles, val_titles, train_summaries, val_summaries, train_ratings, val_ratings = train_test_split(
        train_val_titles, train_val_summaries, train_val_ratings, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # 初始化tokenizer和模型配置
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    config = BertConfig.from_pretrained('bert-base-chinese')
    config.num_hidden_layers = 6  # 减少层数以节省内存

    # 创建数据集和数据加载器
    train_dataset = AnimeDataset(train_titles, train_summaries, train_ratings, tokenizer)
    val_dataset = AnimeDataset(val_titles, val_summaries, val_ratings, tokenizer)
    test_dataset = AnimeDataset(test_titles, test_summaries, test_ratings, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 初始化模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = BertModel.from_pretrained('bert-base-chinese', config=config)
    model = AnimeRatingModel(bert_model).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # 添加权重衰减
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # 训练循环
    num_epochs = 20  # 增加 epoch 数量，让早期停止机制起作用
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'model4_upgrade_anime_rating_model.pth')
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load('model4_upgrade_anime_rating_model.pth'))
    test_loss, test_accuracy, test_accuracy_05, predictions, true_ratings = test(model, test_loader, criterion, device)
    print(f'\nFinal Test Loss: {test_loss:.4f}')
    print(f'Final Test Accuracy (within 1 point): {test_accuracy:.4f}')
    print(f'Final Test Accuracy (within 0.5 points): {test_accuracy_05:.4f}')

if __name__ == '__main__':
    main()

# Epoch 1/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [04:05<00:00,  2.64it/s, loss=1.0426]
# Train Loss: 1.6898, Val Loss: 0.6389, Val Accuracy: 0.8220

# Epoch 2/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [04:00<00:00,  2.70it/s, loss=0.4743]
# Train Loss: 0.6640, Val Loss: 0.6086, Val Accuracy: 0.8445

# Epoch 3/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:42<00:00,  3.99it/s, loss=0.2792]
# Train Loss: 0.4049, Val Loss: 0.3416, Val Accuracy: 0.9256

# Epoch 4/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:47<00:00,  3.88it/s, loss=0.1245]
# Train Loss: 0.2504, Val Loss: 0.3036, Val Accuracy: 0.9463

# Epoch 5/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:58<00:00,  3.64it/s, loss=0.1702]
# Train Loss: 0.1906, Val Loss: 0.2242, Val Accuracy: 0.9596

# Epoch 6/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:36<00:00,  4.14it/s, loss=0.2184]
# Train Loss: 0.1687, Val Loss: 0.2731, Val Accuracy: 0.9573

# Epoch 7/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:31<00:00,  4.28it/s, loss=0.2387]
# Train Loss: 0.1437, Val Loss: 0.2410, Val Accuracy: 0.9619

# Epoch 8/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:32<00:00,  4.28it/s, loss=0.0917]
# Train Loss: 0.1365, Val Loss: 0.1877, Val Accuracy: 0.9677

# Epoch 9/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:31<00:00,  4.28it/s, loss=0.0847]
# Train Loss: 0.1195, Val Loss: 0.2937, Val Accuracy: 0.9599

# Epoch 10/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:31<00:00,  4.29it/s, loss=0.0792]
# Train Loss: 0.1122, Val Loss: 0.1965, Val Accuracy: 0.9631

# Epoch 11/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:31<00:00,  4.28it/s, loss=0.1212]
# Train Loss: 0.1065, Val Loss: 0.1942, Val Accuracy: 0.9657

# Epoch 12/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:31<00:00,  4.28it/s, loss=0.1194]
# Train Loss: 0.0865, Val Loss: 0.2254, Val Accuracy: 0.9657

# Epoch 13/20
# Training: 100%|█████████████████████████████████████████████████████| 650/650 [02:31<00:00,  4.29it/s, loss=0.1282]
# Train Loss: 0.0798, Val Loss: 0.1929, Val Accuracy: 0.9674
# Early stopping triggered after 13 epochs
# Testing: 100%|███████████████████████████████████████████████████████████████████| 217/217 [00:18<00:00, 11.50it/s]

# Test Loss: 0.2054
# Test Accuracy (predictions within 1 point): 0.9605
# Test Accuracy (predictions within 0.5 points): 0.8454

# Final Test Loss: 0.2054
# Final Test Accuracy (within 1 point): 0.9605
# Final Test Accuracy (within 0.5 points): 0.8454