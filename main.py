import torch
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from transformer_model import Transformer, train, evaluate
from torchtext.vocab import build_vocab_from_iterator
from tensorboardX import SummaryWriter

import spacy

# 加载spacy分词器
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")


# 定义分词函数
def tokenize(text, spacy_tokenizer):
    if isinstance(text, str):
        return [tok.text.lower() for tok in spacy_tokenizer(text)]
    else:
        # 如果输入不是字符串类型，则返回空列表或者其他你认为合适的处理方式
        return []


# 加载数据集
train_data, valid_data, test_data = Multi30k(
    split=("train", "valid", "test"), language_pair=("en", "de")
)

# 构建词汇表
SRC = build_vocab_from_iterator(
    map(lambda x: tokenize(x[0], spacy_en.tokenizer), train_data),
    specials=["<unk>", "<pad>", "<bos>", "<eos>"],
)
TRG = build_vocab_from_iterator(
    map(lambda x: tokenize(x[1], spacy_de.tokenizer), train_data),
    specials=["<unk>", "<pad>", "<bos>", "<eos>"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32


# 定义 DataLoader 的整理函数
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sent, trg_sent in batch:
        # 将源句子和目标句子中的词转换为索引，并确保它们是整数类型
        src_tokens = [SRC[token] for token in tokenize(src_sent, spacy_en.tokenizer)]
        trg_tokens = [TRG[token] for token in tokenize(trg_sent, spacy_de.tokenizer)]
        # # 原文
        # print(src_sent, trg_sent)
        # # 映射到词典空间
        # print(src_tokens, trg_tokens)
        # # 打印句子长度
        # print(len(src_batch), len(trg_tokens))
        src_batch.append(torch.tensor(src_tokens, dtype=torch.long))
        trg_batch.append(torch.tensor(trg_tokens, dtype=torch.long))

    # 获取最大的源句子长度和目标句子长度
    max_src_len = max(len(src) for src in src_batch)
    max_trg_len = max(len(trg) for trg in trg_batch)
    max_len = max(max_src_len, max_trg_len)

    # 使用最大长度填充句子
    src_batch = [
        torch.cat((src, torch.full((max_len - len(src),), SRC["<pad>"], dtype=torch.long)))
        for src in src_batch
    ]
    trg_batch = [
        torch.cat((trg, torch.full((max_len - len(trg),), TRG["<pad>"], dtype=torch.long)))
        for trg in trg_batch
    ]

    print("Batch shapes - src:", torch.stack(src_batch).shape, "trg:", torch.stack(trg_batch).shape)
    
    return torch.stack(src_batch), torch.stack(trg_batch)


# 创建 DataLoader
train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 模型参数
INPUT_DIM = len(SRC)
OUTPUT_DIM = len(TRG)
print("INPUT_DIM", INPUT_DIM, "OUTPUT_DIM", OUTPUT_DIM)
SRC_PAD_IDX = SRC["<pad>"]
TRG_PAD_IDX = TRG["<pad>"]

# 初始化模型、优化器和损失函数
model = Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX).to(device)
optimizer = optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# TensorBoardX SummaryWriter
writer = SummaryWriter()

# 训练参数
N_EPOCHS = 100
CLIP = 1

# 训练模型
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, device)
    print(f"Epoch: {epoch+1:02}")
    print(f"\t训练损失: {train_loss:.3f}")

    valid_loss = evaluate(model, valid_iterator, criterion, device)
    print(f"\t验证损失: {valid_loss:.3f}")

    # 写入TensorBoard
    writer.add_scalar("Train Loss", train_loss, epoch)
    writer.add_scalar("Valid Loss", valid_loss, epoch)
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    # 每10个epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

writer.close()
