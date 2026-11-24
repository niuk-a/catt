
import torch.nn as nn
from transformer import *

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, learnable_pos_emb=True):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               padding_idx=src_pad_idx,
                               learnable_pos_emb=learnable_pos_emb)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               padding_idx=trg_pad_idx,
                               learnable_pos_emb=learnable_pos_emb)

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, src, trg, cu_seqlens=None, max_seqlen=None, positions=None):
        if cu_seqlens is not None:
            # Source and shifted target have equal lengths in teacher-forced training.
            seq_info = (cu_seqlens, max_seqlen)
            enc_src = self.encoder(x=src, seq_info=seq_info, positions=positions)
            return self.decoder(trg=trg, enc_src=enc_src, trg_info=seq_info,
                                src_info=seq_info, positions=positions)
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) & \
                   self.make_no_peak_mask(trg, trg)
        enc_src = self.encoder(x=src, s_mask=src_mask)
        output = self.decoder(trg=trg, enc_src=enc_src,
                              trg_mask=trg_mask, src_mask=src_trg_mask)
        return output

    def make_pad_mask(self, q, k, q_pad_idx, k_pad_idx):
        return q.ne(q_pad_idx)[:, None, :, None] & k.ne(k_pad_idx)[:, None, None, :]

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        # len_q x len_k
        mask = torch.ones(len_q, len_k, dtype=torch.bool, device=q.device).tril()
        return mask

