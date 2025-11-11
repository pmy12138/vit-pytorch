# hybrid_model.py
import torch
import torch.nn as nn
from vit_model import Block, PatchEmbed, VisionTransformer, _init_vit_weights  # 只取Block等组件
from resnet_model_GN_WS import resnet18_gn_ws, ResNet  # resnet18_gn_ws factory or ResNet class

class HybridViT(nn.Module):
    def __init__(self,
                 cnn_backbone='resnet18_gn_ws',
                 pretrained_backbone=False,
                 cnn_out_layer='layer3',   # 从哪个层取feature_map: layer3 -> spatial ~8x8 for CIFAR32
                 embed_dim=128,
                 depth=8,
                 num_heads=4,
                 mlp_ratio=4.0,
                 num_classes=10,
                 drop_ratio=0.2,
                 attn_drop_ratio=0.2,
                 drop_path_ratio=0.1):
        super().__init__()
        # 1) CNN backbone (不带 classifier head)
        # 使用你提供的 resnet18_gn_ws(include_top=False) 或者 ResNet(...) 自行实例化
        # 注意：resnet18_gn_ws 工厂在 model_GN_WS.py 中定义，默认 include_top=True -> 我们会改为 False
        # 如果你的工厂不接受 include_top 参数，可直接使用 ResNet(...) 构建
        self.cnn = resnet18_gn_ws(num_classes=10, include_top=False)  # include_top=False -> 返回 feature map
        # （若工厂不支持 include_top 参数，请改为 ResNet(...) 直接创建）

        # 如果不想加载预训练，跳过；否则 load_state_dict(...)
        if not pretrained_backbone:
            # 保持随机初始化
            pass

        # 2) 选取 CNN 输出通道数（由于 model_GN_WS 的 ResNet layer3 输出通道为 256 for ResNet18）
        # 我们对任意输出 C 使用一个 1x1 conv 将 C -> embed_dim
        # 先做一次 dummy forward 的 shape 探测，或者默认知道 resnet18 layer3 输出 256 channels
        self.cnn_proj = nn.Conv2d(in_channels=256, out_channels=embed_dim, kernel_size=1, stride=1)

        # 3) Transformer part: we reuse Block (attention+mlp) from vit_model
        self.embed_dim = embed_dim
        self.num_tokens = 1  # no distill token
        # class token + pos embedding created later once we know num_patches (dynamic)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 默认初始化一个 placeholder pos_embed，以便加载权重时不会报错
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, embed_dim))  # 65=8*8+1，对CIFAR10足够
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self._pos_embed_initialized = True

        # transformer blocks
        norm_layer = nn.LayerNorm
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  qk_scale=None, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i], norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)


    def forward(self, x):
        # x: [B, 3, 32, 32]
        # CNN forward until layer3
        # The resnet forward currently returns after layer4; to get intermediate layer output, we need a small helper.
        # For simplicity assume we have modified resnet to provide a method `forward_features_until(layername)`
        # If not, we can call sequentially:
        B = x.shape[0]

        # Manual forward through resnet to get layer3 output (adapt if your resnet API differs)
        # Using the structure of model_GN_WS.ResNet:
        # conv1 -> gn1 -> relu -> layer1 -> layer2 -> layer3 -> layer4 ...
        out = self.cnn.conv1(x)
        out = self.cnn.gn1(out)
        out = self.cnn.relu(out)
        out = self.cnn.layer1(out)
        out = self.cnn.layer2(out)
        out = self.cnn.layer3(out)  # we stop here (spatial ~ 8x8 for CIFAR-10)

        # project channels -> embed_dim
        proj = self.cnn_proj(out)   # [B, embed_dim, Hc, Wc]
        B, C, Hc, Wc = proj.shape
        num_patches = Hc * Wc

        # create pos_embed if not inited
        if self.pos_embed.shape[1] != num_patches + self.num_tokens:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim).to(proj.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # flatten spatial -> tokens
        x_flat = proj.flatten(2).transpose(1, 2)  # [B, Hc*Wc, embed_dim]

        # prepend cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,embed_dim]
        x_tokens = torch.cat((cls_tokens, x_flat), dim=1)  # [B, 1+num_patches, embed_dim]

        x_tokens = self.pos_drop(x_tokens + self.pos_embed)

        x_tokens = self.blocks(x_tokens)
        x_tokens = self.norm(x_tokens)
        cls_out = x_tokens[:, 0]
        logits = self.head(cls_out)

        return logits
