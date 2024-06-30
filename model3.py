import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
from sklearn.model_selection import KFold
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

def train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler):
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

# 设置交叉验证参数
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建数据集
    dataset = AnimeDataset(subjects, titles, summaries, ratings, tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    fold_train_losses = []
    fold_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            dataset, 
            batch_size=16,
            sampler=train_subsampler
        )
        val_loader = DataLoader(
            dataset,
            batch_size=16, 
            sampler=val_subsampler
        )
        
        # 初始化模型、损失函数和优化器
        config = BertConfig.from_pretrained('bert-base-chinese')  
        config.num_hidden_layers = 6
        bert_model = BertModel.from_pretrained('bert-base-chinese', config=config)
        model = AnimeRatingModel(bert_model).to(device)
        criterion = nn.MSELoss()  
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        
        # 训练循环
        for epoch in range(50):  # 每个fold内最多训练50个epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                print(f"Early stopping triggered in fold {fold+1}")
                break
        
        fold_train_losses.append(train_loss)
        fold_val_losses.append(best_val_loss)
                
        print(f"Fold {fold+1} finished. Best Val Loss: {best_val_loss:.4f}")
        
    print(f"\n{n_splits}-Fold Cross Validation finished.")
    print(f"Average Train Loss: {sum(fold_train_losses)/len(fold_train_losses):.4f}")  
    print(f"Average Val Loss: {sum(fold_val_losses)/len(fold_val_losses):.4f}")

if __name__ == '__main__':
    main()