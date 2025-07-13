import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleDotAttention(nn.Module):
    def __init__(self, temperature, dropout_rate = 0.1):
        super().__init__()
        self.temp = temperature
        self.softmax_dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v):
        attn_weights = torch.matmul(q/self.temp, k.transpose(-2,-1))
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.softmax_dropout(attn_probs)
        context_layer = torch.matmul(attn_probs, v)
        return context_layer, attn_weights

class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.num_heads
        self.d_model = config.hidden_dim
        self.d_k = int(config.hidden_dim/self.n_head)
        
        self.wq = nn.Linear(in_features = self.d_model, out_features = self.n_head*self.d_k)
        self.wk = nn.Linear(in_features = self.d_model, out_features = self.n_head*self.d_k)
        self.wv = nn.Linear(in_features = self.d_model, out_features = self.n_head*self.d_k)

        self.attention_layer = ScaleDotAttention(temperature = self.d_k**0.5, dropout_rate=config.attn_dropout_rate)

        self.fc = nn.Linear(in_features = config.hidden_dim, out_features = config.hidden_dim)
        self.fc_dropout = nn.Dropout(config.attn_dropout_rate)

    def forward(self, z: torch.Tensor):
        torch._assert(z.dim() == 3, "Expected (batch_size, seq_length, hidden_dim) got {}".format(z.shape))
        z_shape = z.shape
        q = self.wq(z).view(z_shape[0], z_shape[1], self.n_head, self.d_k)
        k = self.wk(z).view(z_shape[0], z_shape[1], self.n_head, self.d_k)
        v = self.wv(z).view(z_shape[0], z_shape[1], self.n_head, self.d_k)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        context_layer, n_weights = self.attention_layer(q,k,v)
        # context_layer: N x N_head x sq x dk
        # weights: N x N_head x sq x sq        

        avg_weights = torch.sum(n_weights, dim = 1) / self.n_head
        # avg_weights: N x sq x sq

        context_layer = context_layer.permute(0,2,1,3)
        context_layer = context_layer.contiguous.view(z_shape[0], z_shape[1], -1)
        attn_output = self.fc(context_layer)
        attn_output = self.fc_dropout(attn_output)
        # attn_output: N x sq x hd
        return attn_output, avg_weights
    
class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        np = int((config.img_size * config.img_size) / (config.patch_size * config.patch_size))
        self.d_model = config.hidden_dim
        self.patch_embedding = nn.Conv2d(in_channels = 3, out_channels = config.hidden_dim, kernel_size=config.patch_size, stride = config.patch_size)
        self.positional_embedding = nn.Parameter(torch.zeros([np, config.hidden_dim]))
        self.dropout = nn.Dropout(config.embed_dropout_rate)
    
    def forward(self, x):
        x = self.patch_embedding(x)
        # N x HD x W/P x H/P
        x = x.flatten(2)
        # N x HD x NP
        x = x.transpose(-1, -2)
        # N x NP x HD
        x = x + self.positional_embedding
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.hidden_dim
        self.fc1 = nn.Linear(in_features = config.hidden_dim, out_features = config.mlp_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features = config.mlp_dim, out_features = config.hidden_dim)
        self.dropout = nn.Dropout(config.mlp_dropout_rate)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc1.bias)
        nn.init.uniform_(self.fc2.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_layer = MultiheadAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.mlp_layer = MLP(config)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, x):
        h = x
        x, weights = self.attn_layer(x)
        x = self.ln1(x + h)

        h = x
        x = self.mlp_layer(x)
        x = self.ln2(x + h)
        return x, weights

class PatchAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_embed = Embedding(config)
        self.patch_size = config.patch_size
        self.layers = nn.ModuleList()
        for _ in range(config.number_layers):
            layer = Block(config)
            self.layers.append(layer)
        self.softmax_dropout = nn.Dropout(config.pam_dropout_rate)
        self.transconvWeights = self.tranconvWeights = torch.ones((1, 3, config.patch_size, config.patch_size))
    
    def forward(self, imgs, delta):
        nb, _, h, w = imgs.shape
        nh = h//self.patch_size
        nw = w//self.patch_size
        x = self.input_embed(imgs)
        for layer in self.layers:
            x, weights = layer(x)
        attn_scores = torch.sum(weights, dim = -1)
        attn_probs = F.softmax(attn_scores, dim=-1).view(nb, 1, nh, nw)
        attn_probs = self.softmax_dropout(attn_probs)
        attn_probs = F.conv_transpose2d(attn_probs, self.tranconvWeights, stride=self.patch_size, padding=0)
        delta *= attn_probs
        adv_imgs = imgs+delta
        adv_imgs = adv_imgs.clamp(0,1)
        return adv_imgs
