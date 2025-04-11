import torch

class SkelEmbed(torch.nn.Module):
    def __init__(self, embedding_dim=256, emb_img_size=(32,32,4), layer_num=4, transformer_ff_dim=512):
        super(SkelEmbed, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb_img_size = emb_img_size
        self.layer_num = layer_num
        self.transformer_ff_dim = transformer_ff_dim
 
        # in_pointwise_feedforward (6->embedding_dim)
        # attach CLS token with learnable embedding_dim parameters
        # trasnformer_encoder (embedding_dim->embedding_dim)
        # out_pointwise_feedforward (embedding_dim->emb_img_size(x*y*c))
 
        self.in_pointwise_feedforward = torch.nn.Linear(6, self.embedding_dim)
 
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.embedding_dim))
 
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=1,
                dim_feedforward=self.transformer_ff_dim,
                dropout=0.1
            ),
            num_layers=self.layer_num
        )
 
        self.out_pointwise_feedforward = torch.nn.Linear(self.embedding_dim, self.emb_img_size)
 
    def forward(self, x):
        # TODO: handle mirrored edge representations inside this call
        # TODO: padding mask for transformer
        # x: (batch, seq, 6, 2)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size, seq_len, -1)
        x = self.in_pointwise_feedforward(x) # (batch, seq, embedding_dim)
        cls_token = self.cls_token.repeat(batch_size, 1, 1) # (batch, 1, embedding_dim)
        x = torch.cat([cls_token, x], dim=1) # (batch, seq+1, embedding_dim)
        x = x.permute(1, 0, 2) # (seq+1, batch, embedding_dim)
        x = self.transformer_encoder(x) # (seq+1, batch, embedding_dim)
        x = x.permute(1, 0, 2) # (batch, seq+1, embedding_dim)
 
        cls_emb = x[:, 0, :] # (batch, emb_img_size)
        cls_emb = self.out_pointwise_feedforward(cls_emb) # (batch, emb_img_size)
        img_emb = cls_emb.view(batch_size, *self.emb_img_size) # (batch, x, y, c)
 
        return img_emb