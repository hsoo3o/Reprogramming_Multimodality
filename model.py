import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open_clip
from typing import Optional, Tuple


class OpenCLIP_reprogrammed(nn.Module):
    def __init__(self, configs):
        super(OpenCLIP_reprogrammed, self).__init__()

        self.source_type = configs['source_type']
        self.clip_model,_, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.clip_dim = configs['source']['embed_dim']

        self.pool_type = configs['source']['pool_type']
        self.text_pool_type =  configs['source']['text_pool_type']
        self.target_model_proj = self.clip_model.visual.proj
        self.target_model_text_projection = self.clip_model.text_projection



        self.source_model_img = self.clip_model.visual

        self.source_model_transformer_img = self.clip_model.visual.transformer
        self.source_model_ln_post_img = self.clip_model.visual.ln_post
        
        self.source_model_txt = self.clip_model.encode_text
        self.source_model_transformer_txt = self.clip_model.transformer
        self.source_model_ln_final_txt = self.clip_model.ln_final
        

        source_len = (configs['source']['input_resolution'] - configs['source']['patch_size']) / configs['source']['patch_size'] + 1
        self.source_len = int(source_len* source_len + 1)
        
        depth_len = (configs['depth']['input_resolution'] - configs['depth']['patch_size']) / configs['depth']['patch_size'] + 1
        self.depth_len = int( depth_len * depth_len + 1)

        audio_freq_len = (configs['audio']['audio_num_mel_bins'] - configs['audio']['audio_kernel_size']) / configs['audio']['audio_stride'] + 1
        audio_time_len = (configs['audio']['audio_target_len'] - configs['audio']['audio_kernel_size']) / configs['audio']['audio_stride'] + 1
        self.audio_len = int(int(audio_freq_len) * int(audio_time_len) + 1)


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 /0.07))

        self.modality_tokenizers = self._create_modality_tokenizers(
            configs['depth']['patch_size'],
            configs['source']['embed_dim'],
            configs['source']['text_emb_dim'],
            configs['depth']['vid_frame_num'],
            configs['audio']['audio_num_mel_bins'],
            configs['audio']['audio_target_len'],
            configs['audio']['audio_kernel_size'],
            configs['audio']['audio_stride'],
        )
        self.modality_reprogramming_layers = self._create_modality_reprogramming_layer(
            emb_dim=configs['source']['embed_dim'],
            num_heads=configs['reprogram']['reprogram_num_heads'],
            dropout=configs['reprogram']['reprogram_dropout']
        )

        self.reprogram_result = None
        self.reprogram_att_weight = None
        self.mid_logit = None
        
    
    @property
    def dtype(self):
        return self.clip_model.visual.conv1.weight.dtype

    def reprogram_forward(self, target_type, source, target):
        chg_dims = len(target.shape)
        if chg_dims > 4:
            B, S = target.shape[:2]  
            target = target.reshape(
                B * S, *target.shape[2:]
            )        

        chg_dims_s = len(source.shape)
        if chg_dims_s > 4:
            Bs, Ss = source.shape[:2]  
            source = source.reshape(
                Bs * Ss, *source.shape[2:]
            )

        if self.source_type =='text':
            source_tokenized, attn_mask = self.modality_tokenizers[self.source_type](source)
            target_tokenized = self.modality_tokenizers[target_type](target)

            target_tokenized = target_tokenized.permute(0, 2, 1)
            reprogrammed = self.modality_reprogramming_layers[target_type](target_tokenized)
            self.reprogram_result = reprogrammed
            
            # mid logit
            source_normed = source_tokenized / source_tokenized.norm(dim=1, keepdim=True)
            target_normed = reprogrammed / reprogrammed.norm(dim=1, keepdim=True)

            source_normed = source_normed.mean(dim=1)
            target_normed = target_normed.mean(dim=1)

            logit_scale = self.logit_scale.exp()
            self.mid_logit = logit_scale * source_normed @ target_normed.t()
            # 

            x = self.source_model_transformer_txt(reprogrammed.type(self.dtype), attn_mask=attn_mask.to(target.device))

            x = self.source_model_ln_final_txt(x).type(self.dtype)
            pooled, _ = text_global_pool(x, target, self.text_pool_type)

            pooled = pooled @ self.target_model_text_projection

            if chg_dims_s > 4:
                pooled = pooled.reshape(Bs, Ss, -1)
                pooled = pooled.mean(dim=1)
            

        else:
            source_tokenized = self.modality_tokenizers[self.source_type](source)
            target_tokenized = self.modality_tokenizers[target_type](target)

            target_tokenized = target_tokenized.permute(0, 2, 1)
            reprogrammed = self.modality_reprogramming_layers[target_type](target_tokenized)
            self.reprogram_result = reprogrammed

            # mid logit
            source_normed = source_tokenized / source_tokenized.norm(dim=1, keepdim=True)
            target_normed = reprogrammed / reprogrammed.norm(dim=1, keepdim=True)

            source_normed = source_normed.mean(dim=1)
            target_normed = target_normed.mean(dim=1)
            
            logit_scale = self.logit_scale.exp()
            self.mid_logit = logit_scale * source_normed @ target_normed.t()
            # 

            x = self.source_model_transformer_img(reprogrammed.type(self.dtype))
            x = self.source_model_ln_post_img(x)
            
            pooled, tokens = self._global_pool(x)

            if self.target_model_proj is not None:
                pooled = pooled @ self.target_model_proj

            if chg_dims_s > 4:
                pooled = pooled.reshape(Bs, Ss, -1)
                pooled = pooled.mean(dim=1)

        

        return pooled
    

    
    def target_forward(self, target_type, target):
        
        chg_dims = len(target.shape)
        if chg_dims > 4:
            B, S = target.shape[:2]  
            target = target.reshape(
                B * S, *target.shape[2:]
            )

        

        if target_type =='text':  
            target_tokenized, attn_mask = self.modality_tokenizers[target_type](target)
            x = self.source_model_transformer_txt(target_tokenized.type(self.dtype), attn_mask=attn_mask.to(target.device))

            x = self.source_model_ln_final_txt(x).type(self.dtype)
            pooled, _ = text_global_pool(x, target, self.text_pool_type)

            pooled = pooled @ self.target_model_text_projection
            
        elif target_type == 'image' or target_type == 'depth':
            target_tokenized = self.modality_tokenizers[target_type](target)

            x = self.source_model_transformer_img(target_tokenized.type(self.dtype))
            x = self.source_model_ln_post_img(x)
            
            pooled, tokens = self._global_pool(x)

            if self.target_model_proj is not None:
                pooled = pooled @ self.target_model_proj

            if chg_dims > 4:
                pooled = pooled.reshape(B, S, -1)
                pooled = pooled.mean(dim=1)
            

        else:
            target_tokenized = self.modality_tokenizers[target_type](target)
            target_tokenized = target_tokenized.permute(0, 2, 1)
            reprogrammed = self.modality_reprogramming_layers[target_type](target_tokenized)
            self.reprogram_result = reprogrammed

            x = self.source_model_transformer_img(reprogrammed.type(self.dtype))

            x = self.source_model_ln_post_img(x)
            
            pooled, tokens = self._global_pool(x)

            if self.target_model_proj is not None:
                pooled = pooled @ self.target_model_proj

            if chg_dims > 4:
                pooled = pooled.reshape(B, S, -1)
                pooled = pooled.mean(dim=1)
            
        
        return pooled


    def forward(self, 
                targets: Optional[dict] = None,
                source: Optional[torch.Tensor] = None
                ):
        
        outputs = {}
        if source is not None and targets is not None:
            
            if self.source_type == 'image':
                source_features = self.source_model_img(source.type(self.dtype))
            elif self.source_type == 'text':
                source_features = self.source_model_txt(source.type(self.dtype))
            outputs['source_'+ self.source_type] = source_features

            for target_type, target_value in targets.items():
                target_features = self.reprogram_forward(target_type, source, target_value)
                outputs[target_type] = target_features

            return outputs  
        elif targets is not None:
            for target_type, target_value in targets.items():
                target_features = self.target_forward(target_type, target_value)
                outputs[target_type] = target_features
        
        return outputs



    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens



    def _create_modality_tokenizers(self,
                                    patch_size=32,
                                    vision_embed_dim = 512,
                                    text_emb_dim = 512,
                                    vid_frame_num = 2,
                                    audio_num_mel_bins = 128,
                                    audio_target_len = 204,
                                    audio_kernel_size = 32,
                                    audio_stride = 10,
                                    ):

        rgb_tokenizer = VisisualTokenizer(
                in_channels = 3,
                input_resolution = 224,
                patch_size = patch_size,
                embed_dim = vision_embed_dim
            )

        depth_tokenizer =  VisisualTokenizer(
                in_channels = 1,
                input_resolution = 224,
                patch_size = patch_size,
                embed_dim = vision_embed_dim
            )

        video_tokenizer =  VideoTokenizer(
                in_channels = 3,
                input_resolution = 224,
                patch_size = patch_size,
                frame_num = vid_frame_num,
                embed_dim = vision_embed_dim
            )

        text_tokenizer = TextTokenizer(
            vocab_size = 49408,
            context_length = 77,
            embed_dim = text_emb_dim,
            dtype = self.dtype
        )

        audio_tokenizer = AudioTokenizer(
            in_channels = 1,
            input_resolution = [1, audio_num_mel_bins, audio_target_len],
            patch_size = audio_kernel_size,
            stride_size = audio_stride,
            embed_dim = vision_embed_dim
        )

        modality_preprocessors = {
            'image': rgb_tokenizer,
            'video': video_tokenizer,
            'text': text_tokenizer,
            'audio': audio_tokenizer,
            'depth': depth_tokenizer,
        }

        return nn.ModuleDict(modality_preprocessors)

    def _create_modality_reprogramming_layer(self, emb_dim, num_heads, dropout):

        audio_reprogramming = Reprogramming_attention(
            emb_dim=emb_dim,
            clip_dim=self.source_len,
            num_heads=num_heads,
            dropout=dropout
        )

        depth_reprogramming = Reprogramming_attention(
            emb_dim=emb_dim,
            clip_dim=self.source_len,
            num_heads=num_heads,
            dropout=dropout
        )



        # audio_reprogramming = Reprogramming_linear(
        #     clip_dim = self.source_len,
        #     modal_dim = self.audio_len,
        # )

        # depth_reprogramming = Reprogramming_linear(
        #     clip_dim = self.source_len,
        #     modal_dim = self.depth_len,
        # )

        # imu_reprogramming = Reprogramming_linear(
        #     clip_dim = self.source_len,
        #     modal_dim = self.imu_len,
        # )


        modality_reprogram = {
            'audio': audio_reprogramming,
            'depth': depth_reprogramming,
        }

        return nn.ModuleDict(modality_reprogram)



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

class VisisualTokenizer(nn.Module):
    def __init__(self, in_channels: int, input_resolution: int, patch_size: int, embed_dim: int, patch_dropout: float = 0.):
        super().__init__()

        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)



        scale = embed_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, embed_dim))
        self.ln_pre = LayerNorm(embed_dim)
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, embed_dim, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, embed_dim, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, embed_dim]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        
        # pathce dropdout
        x = self.patch_dropout(x)

        x = self.ln_pre(x)

        return x
    

   
class VideoTokenizer(nn.Module):
    def __init__(self, in_channels: int, input_resolution: int, patch_size: int, frame_num:int, embed_dim: int, patch_dropout: float = 0.):
        super().__init__()

        self.input_resolution = input_resolution
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=(frame_num, patch_size, patch_size),  
            stride=(frame_num, patch_size, patch_size), 
            bias=False
        )


        scale = embed_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, embed_dim))
        self.ln_pre = LayerNorm(embed_dim)
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()



    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [batch, embed_dim, frames, grid, grid]
        x = x.flatten(2)
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, embed_dim, grid ** 2]
        x = x.permute(0, 2, 1)
        # x = x.flatten(2).transpose(1, 2)

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        
        # pathce dropdout
        x = self.patch_dropout(x)

        x = self.ln_pre(x)

        return x

class TextTokenizer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embed_dim: int, dtype):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, embed_dim))
        self.dtype = dtype
        self.causal_mask = None


    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, context length, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        self.causal_mask = build_causal_attention_mask(self.context_length)
        self.register_buffer('attn_mask',self.causal_mask, persistent=False)

        return x, self.attn_mask

def build_causal_attention_mask(context_length):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length, requires_grad=False)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask
 
def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens  

class AudioTokenizer(nn.Module):
    def __init__(self, in_channels: int, input_resolution: int, patch_size: int, stride_size:int, embed_dim: int, patch_dropout: float = 0.):
        super().__init__()

        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=stride_size, bias=False)
        self.norm_layer = LayerNorm(embed_dim)


        scale = embed_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))

        num_patches_height = (input_resolution[1] - patch_size) // stride_size + 1
        num_patches_width = (input_resolution[2] - patch_size) // stride_size + 1
        num_patches = num_patches_height * num_patches_width

        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, embed_dim))
        self.ln_pre = LayerNorm(embed_dim)
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, embed_dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, embed_dim, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, embed_dim]
        x = self.norm_layer(x)

        
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        
        # pathce dropdout
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        return x


class ImuTokenizer(nn.Module):
    def __init__(self,  in_channels: int, input_resolution: int, patch_size: int, embed_dim: int, patch_dropout: float = 0.):
        super().__init__()

        self.input_resolution = input_resolution

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm_layer = LayerNorm(embed_dim)

        scale = embed_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution[1] // patch_size) + 1, embed_dim))
        self.ln_pre = LayerNorm(embed_dim)
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
    
    def forward(self, x):
        print('imu임베딩 확인')
        x = x.permute(0, 2, 1)  # [batch_size, num_features, seq_length] -????????????????????
        x = self.conv1(x) # [batch_size, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # [batch_size, num_patches, embed_dim]
        
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        
        # pathce dropdout
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        return x


class Reprogramming_attention(nn.Module):
    def __init__(self, emb_dim, clip_dim, num_heads, dropout):
        super(Reprogramming_attention, self).__init__()
        self.clip_dim = clip_dim
        self.num_heads = num_heads

        # Learnable Query for clip_dim
        self.query = nn.Parameter(torch.randn(1, clip_dim, emb_dim))

        # Multihead Attention
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        """
        x: [batch_size, emb_dim, modal_dim]
        Returns:
        output: [batch_size, emb_dim, clip_dim]
        """
        batch_size = x.size(0)

        # Reshape input for attention: [modal_dim, batch_size, emb_dim]
        x = x.permute(2, 0, 1)

        # Expand Query to batch size: [clip_dim, batch_size, emb_dim]
        query = self.query.expand(batch_size, -1, -1).permute(1, 0, 2)

        # Attention computation
        attn_output, _ = self.attention(query, x, x)  # [clip_dim, batch_size, emb_dim]

        # Reshape back to [batch_size, emb_dim, clip_dim]
        output = attn_output.permute(1, 0, 2)
        return output
    


class Reprogramming_linear(nn.Module):
    def __init__(self, clip_dim, modal_dim):
        super(Reprogramming_linear, self).__init__()
        # Learnable Linear Layer
        self.linear = nn.Linear(modal_dim, clip_dim)

    def forward(self, x):
        """
        x: [batch_size, emb_dim, modal_dim]
        Returns:
        output: [batch_size, emb_dim, clip_dim]
        """
        # Permute to apply Linear on modal_dim
        x = self.linear(x)  # [batch_size, emb_dim, clip_dim]
        x = x.permute(0, 2, 1)  # [batch_size, clip_dim, emb_dim]
        return x
# patch output: b, seq_len(=clip_dim), emb_dim