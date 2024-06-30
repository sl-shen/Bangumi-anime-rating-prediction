# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def train(model, train_loader, optimizer, criterion, device, accumulation_steps=2):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ratings = batch['rating'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), ratings)
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), ratings)
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

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

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # 初始化模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = BertModel.from_pretrained('bert-base-chinese', config=config)
    model = AnimeRatingModel(bert_model).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # 训练循环
    num_epochs = 10
    accumulation_steps = 4  # 梯度累积步数
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device, accumulation_steps)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 在测试集上评估
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f'\nFinal Test Loss: {test_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'anime_rating_model.pth')

if __name__ == '__main__':
    main()

# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:12<00:00, 12.72it/s, loss=1.6234]
# Train Loss: 6.1275, Val Loss: 0.7193

# Epoch 2/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [01:32<00:00,  5.06it/s, loss=0.1264]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:07<00:00, 20.45it/s, loss=1.3369]
# Train Loss: 0.7979, Val Loss: 0.7200

# Epoch 3/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [00:59<00:00,  7.86it/s, loss=0.6851]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:07<00:00, 21.25it/s, loss=1.2171]
# Train Loss: 0.7152, Val Loss: 0.6290

# Epoch 4/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [01:03<00:00,  7.41it/s, loss=0.1742]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:07<00:00, 21.13it/s, loss=1.2692]
# Train Loss: 0.6600, Val Loss: 0.6214

# Epoch 5/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [01:00<00:00,  7.66it/s, loss=0.7685]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:07<00:00, 20.40it/s, loss=1.1313]
# Train Loss: 0.6020, Val Loss: 0.6730

# Epoch 6/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [01:07<00:00,  6.90it/s, loss=1.0578]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:07<00:00, 20.19it/s, loss=1.3224]
# Train Loss: 0.5327, Val Loss: 0.7549

# Epoch 7/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [01:54<00:00,  4.09it/s, loss=0.4215]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:17<00:00,  8.83it/s, loss=1.3429]
# Train Loss: 0.4515, Val Loss: 0.6842

# Epoch 8/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [02:27<00:00,  3.16it/s, loss=0.3928]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:15<00:00, 10.01it/s, loss=1.1609]
# Train Loss: 0.3856, Val Loss: 0.6662

# Epoch 9/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [02:24<00:00,  3.23it/s, loss=0.3256]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:16<00:00,  9.55it/s, loss=1.3843]
# Train Loss: 0.3199, Val Loss: 0.6804

# Epoch 10/10
# Training: 100%|███████████████████████████████████████████████████████████████| 467/467 [02:22<00:00,  3.27it/s, loss=0.2348]
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:18<00:00,  8.58it/s, loss=1.3292]
# Train Loss: 0.2854, Val Loss: 0.6950
# Evaluating: 100%|█████████████████████████████████████████████████████████████| 156/156 [00:18<00:00,  8.63it/s, loss=0.4396]

# Final Test Loss: 0.6922