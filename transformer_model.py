import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        src_pad_idx,
        trg_pad_idx,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.fc_out = nn.Linear(d_model, output_dim)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src, trg):
        print(src.shape, trg.shape)
        # 生成 src_mask, 用于屏蔽 pad 位置的信息, 详细的计算过程会在下面的函数中进行说明
        src_mask = self.make_src_mask(src)
        print("src_mask", src_mask.shape)
        # 生成 trg_mask, 用于屏蔽未来时刻的信息, 详细的计算过程会在下面的函数中进行说明
        trg_mask = self.make_trg_mask(trg)
        print("trg_mask", trg_mask.shape)
        # token 换成 d_model 维的词向量
        src_emb = self.embedding(src)
        print("src_emb", src_emb.shape)
        trg_emb = self.embedding(trg)
        print("trg_emb", trg_emb.shape)
        # 输入到nn.transformer中进行计算
        output = self.transformer(
            src_emb,
            trg_emb,
            src_mask=src_mask,
            tgt_mask=trg_mask,
            src_key_padding_mask=self.src_key_padding_mask(src),
            tgt_key_padding_mask=self.trg_key_padding_mask(trg),
        )
        print("output", output.shape)
        output = self.fc_out(output)
        return output

    # 计算 src_key_padding_mask。这个函数的作用是将 src 中的 pad 位置标记为 True，其他位置标记为 False
    def src_key_padding_mask(self, src):
        return (src == self.src_pad_idx).transpose(0, 1)

    # 计算 trg_key_padding_mask。这个函数的作用是将 trg 中的 pad 位置标记为 True，其他位置标记为 False
    def trg_key_padding_mask(self, trg):
        return (trg == self.trg_pad_idx).transpose(0, 1)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=trg.device)
        ).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask


def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        print("src", src.shape, "trg", trg.shape)
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        # output_dim = output.shape[-1]
        # output = output.reshape(-1, output_dim)
        # trg = trg[1:, :].reshape(-1)
        loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg)
            # output_dim = output.shape[-1]
            # output = output.reshape(-1, output_dim)
            # trg = trg[1:, :].reshape(-1)
            loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)
