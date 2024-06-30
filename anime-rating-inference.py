# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd

class AnimeDataset(Dataset):
    def __init__(self, subjects, titles, summaries, tokenizer, max_length=512):
        self.subjects = subjects
        self.titles = titles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = str(self.subjects[idx]) if not pd.isna(self.subjects[idx]) else ""
        title = str(self.titles[idx]) if not pd.isna(self.titles[idx]) else ""
        summary = str(self.summaries[idx]) if not pd.isna(self.summaries[idx]) else ""

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
        }

class AnimeRatingModel(torch.nn.Module):
    def __init__(self, bert_model, dropout_rate=0.3):
        super(AnimeRatingModel, self).__init__()
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

def predict(model, dataset, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            outputs = model(input_ids, attention_mask)
            prediction = outputs.squeeze().item()
            # 将输出映射到 1-10 的范围
            prediction = max(1, min(10, prediction))
            predictions.append(prediction)
    return predictions

def main():
    # 加载模型配置
    config = BertConfig.from_pretrained('bert-base-chinese')
    config.num_hidden_layers = 6  # 确保与训练时使用的配置相同

    # 初始化 tokenizer 和 BERT 模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese', config=config)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AnimeRatingModel(bert_model).to(device)
    model.load_state_dict(torch.load('anime_rating_model.pth', map_location=device))

    # 准备测试数据
    test_data = [
        {
            "subject": "测试动漫1",
            "title": "亚托莉 -我挚爱的时光-",
            "summary": """在不远的未来，海平面原因不明地急速上升，导致了地表多数都沉入海中。小时候因为事故而失去一条腿的少年斑鸠夏生，厌倦了
都市的生活，移居到了海边的乡村小镇。曾经身为海洋地质学家的祖母留给他的，就只有船、潜艇还有债务。夏生为了取回"失去的未来"，潜海前往
据说沉睡着祖母遗产的海底仓库。在那里，他遇到了一位沉睡在如同棺材一般装置中不可思议的少女"亚托莉"。她是一位构造精密到与人类别无二致
，而又感情丰富的机器人。从海底被打捞起来的亚托莉如是说道。"我想完成主人留给我的最后的命令。在此之前，我会成为夏生先生的腿！"在一个
逐渐沉入海中的平静小镇，少年和机器人少女的难忘夏日就这么开始了。"""
        },
        {
            "subject": "测试动漫2",
            "title": "电锯人",
            "summary": "电锯人是由藤本树创作的日本漫画作品。故事讲述了主人公登地成为恶魔猎人「电锯人」后的种种遭遇。作品于《周刊少年Jump》2018年1号至2020年52号连载第一部，单行本全11卷，在2022年10月12日动画化播出。该作品以独特的人物塑造和剧情发展赢得了广泛好评。"
        },
        {
            "subject": "测试动漫3",
            "title": "2.5次元的诱惑",
            "summary": "我对3D女孩没有兴趣！今天，漫画研究部部长奥村独自一人在自己的俱乐部房间里，喊着他心爱的二次元角色莉莉尔的名字，莉莉尔的名字就映在屏幕的另一边……来到奥村的人是天野莉莉莎，一位3D女孩，她说：“我想成为莉莉尔。”她喜欢漫画中出现的顽皮可爱的女孩“服装”。而且他还是一个和奥村一样爱莉莉尔的御宅族！莉丽莎向奥村透露她的秘密爱好是角色扮演，并向他展示了她收藏的装满角色扮演照片和视频的ROM 。“我……我想做这个！！”Cosplay活动在俱乐部房间开始，只有我们两个人！莉莉莎变身的莉莉尔太真实了，奥村震惊了！？奥村也激动地拿起相机！？他们用真诚和热情面对cosplay“我爱你”所有御宅族的cosplay青春故事。开始！！"
        },
        {
            "subject": "测试动漫4",
            "title": "魔导具师妲莉亚永不妥协",
            "summary": "她最后看到的就是一张堆满文件的桌子。工作了一晚上，他的心脏突然停止了跳动。我想做的事情有很多，但最后却只能瘫倒在办公桌上……她最终在一个存在魔法的世界里度过了第二次生命。她的名字叫达莉亚·罗塞蒂。在她生活的这个世界里，存在着让人们的生活更加便利的“神器”。她的父亲卡罗日常生活懒惰，但却可以被尊为魔法工具制造者。达莉亚很欣赏卡洛制作的魔法工具，并决定自己也成为一名魔法工具制造者。这是她作为魔具制造者的成长故事。"
        },
        {
            "subject": "测试动漫5",
            "title": "【我推的孩子】 第二季",
            "summary": "“谎言是这个娱乐世界的武器，”在当地城市工作的妇产科医生五郎说。有一天，他“最喜欢”的偶像“B小町”爱出现在他面前。她有一个禁忌的秘密……从两人“最糟糕”的相遇开始，命运开始改变。"
        },
        {
            "subject": "测试动漫6",
            "title": "夜樱家的大作战",
            "summary": "「这个家庭非比寻常！？」高中生「朝野太阳」，因为事故而痛失了家人，心灵也因此变得十分封闭。在班上的同学，大家对他的印象都是“超·内向”，唯一能够与之交谈的是他的青梅竹马「夜樱六美」，但是六美却有一个秘密！？她的真实身份是代代相传的间谍家族的当家，还一直受到各种敌人性命的威胁。而且她的哥哥「凶一郎」是最凶狠的间谍，对六美表现出异常的溺爱。太阳发现，自己已经成为凶一郎“排除”的目标，生命岌岌可危……！为了守护六美以及自己的性命安全，太阳采取的手段居然是与六美结婚，成为夜樱家的女婿！？围绕着太阳、六美以及夜樱家前所未闻的「大作战」让人目不转睛！"
        }
    ]

    # 创建测试数据集
    test_dataset = AnimeDataset(
        [item["subject"] for item in test_data],
        [item["title"] for item in test_data],
        [item["summary"] for item in test_data],
        tokenizer
    )

    # 进行预测
    predictions = predict(model, test_dataset, device)

    # 打印结果
    for i, prediction in enumerate(predictions):
        print(f"动漫: {test_data[i]['subject']} - {test_data[i]['title']}")
        print(f"简介: {test_data[i]['summary'][:100]}...")  # 只打印前100个字符
        print(f"预测评分: {prediction:.2f}")
        print()

if __name__ == '__main__':
    main()