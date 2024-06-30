# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class AnimeDataset(Dataset):
    def __init__(self, subjects, titles, summaries, ratings, tokenizer, max_length=512):
        self.subjects = subjects
        self.titles = titles
        self.summaries = summaries
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = str(self.subjects[idx]) if not pd.isna(self.subjects[idx]) else ""
        title = str(self.titles[idx]) if not pd.isna(self.titles[idx]) else ""
        summary = str(self.summaries[idx]) if not pd.isna(self.summaries[idx]) else ""
        rating = float(self.ratings[idx])

        if not subject and not title and not summary:
            text = "无内容"
        else:
            text = f"{subject} {title} {summary}".strip()

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
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), ratings)
            total_loss += loss.item()

    return total_loss / len(val_loader)

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

    subjects = df['subject'].tolist()
    titles = df['title'].tolist()
    summaries = df['subject_summary'].tolist()
    ratings = df['rating'].tolist()

    # 首先分割出测试集
    train_val_subjects, test_subjects, train_val_titles, test_titles, train_val_summaries, test_summaries, train_val_ratings, test_ratings = train_test_split(
        subjects, titles, summaries, ratings, test_size=0.2, random_state=42)

    # 然后将剩余的数据分割为训练集和验证集
    train_subjects, val_subjects, train_titles, val_titles, train_summaries, val_summaries, train_ratings, val_ratings = train_test_split(
        train_val_subjects, train_val_titles, train_val_summaries, train_val_ratings, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # 初始化tokenizer和模型配置
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    config = BertConfig.from_pretrained('bert-base-chinese')
    config.num_hidden_layers = 6  # 减少层数以节省内存

    # 创建数据集和数据加载器
    train_dataset = AnimeDataset(train_subjects, train_titles, train_summaries, train_ratings, tokenizer)
    val_dataset = AnimeDataset(val_subjects, val_titles, val_summaries, val_ratings, tokenizer)
    test_dataset = AnimeDataset(test_subjects, test_titles, test_summaries, test_ratings, tokenizer)

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
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_anime_rating_model.pth')
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load('best_anime_rating_model.pth'))
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f'\nFinal Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()

# Epoch 1/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:59<00:00,  3.92it/s, loss=1.8279]
# Train Loss: 2.8592, Val Loss: 0.6786

# Epoch 2/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:58<00:00,  4.03it/s, loss=0.9422]
# Train Loss: 0.7725, Val Loss: 0.7064

# Epoch 3/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:59<00:00,  3.93it/s, loss=0.4903]
# Train Loss: 0.7038, Val Loss: 0.6247

# Epoch 4/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:56<00:00,  4.17it/s, loss=0.3015]
# Train Loss: 0.6429, Val Loss: 0.8367

# Epoch 5/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:58<00:00,  3.97it/s, loss=0.0025]
# Train Loss: 0.5815, Val Loss: 0.6988

# Epoch 6/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:55<00:00,  4.22it/s, loss=0.2833]
# Train Loss: 0.4734, Val Loss: 0.6120

# Epoch 7/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:58<00:00,  3.97it/s, loss=0.9353]
# Train Loss: 0.3775, Val Loss: 0.6996

# Epoch 8/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:54<00:00,  4.26it/s, loss=0.2354]
# Train Loss: 0.3029, Val Loss: 0.6263

# Epoch 9/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:57<00:00,  4.06it/s, loss=0.2246]
# Train Loss: 0.2516, Val Loss: 0.6514

# Epoch 10/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:56<00:00,  4.13it/s, loss=0.1651]
# Train Loss: 0.1820, Val Loss: 0.6647

# Epoch 11/20
# Training: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:55<00:00,  4.25it/s, loss=0.0410]
# Train Loss: 0.1727, Val Loss: 0.6464
# Early stopping triggered after 11 epochs

# Final Test Loss: 0.6311