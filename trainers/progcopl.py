import os.path as osp
import time
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ProgCoPL',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "progcopl_length": cfg.TRAINER.PROGCOPL.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim).half()
        self.key_proj = nn.Linear(embed_dim, embed_dim).half()
        self.value_proj = nn.Linear(embed_dim, embed_dim).half()
        self.out_proj = nn.Linear(embed_dim, embed_dim).half()

        self.out_dim = nn.Linear(embed_dim, output_dim).half()

        self.attention = ScaledDotProductAttention()

    def forward(self, x, pre_data, mask=None):
        batch_size, seq_length, embed_dim = x.size()

        # Linear projections
        Q = self.query_proj(x)  # (batch_size, seq_length, embed_dim)
        K = self.key_proj(x)  # (batch_size, seq_length, embed_dim)
        V = self.value_proj(x)  # (batch_size, seq_length, embed_dim)

        # Split into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose to get dimensions (batch_size, num_heads, seq_length, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        output, attention_weights = self.attention(Q, K, V, mask=mask)

        # Concatenate heads and put through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(output)
        output = self.out_dim(output)

        with torch.no_grad():
            if pre_data.nelement() != 0:
                pre_data_t = pre_data * 0.1
                output = output * 0.9 + pre_data_t

        return output, attention_weights


class MutualPromptGeneration(nn.Module):
    def __init__(self, ctx_dim, dtype):  # ctx_dim: 512
        super().__init__()
        self.dtype = dtype
        self.multi_t2i = MultiHeadAttention(embed_dim=ctx_dim, output_dim=768, num_heads=8)
        self.multi_i2t = MultiHeadAttention(embed_dim=768, output_dim=ctx_dim, num_heads=8)

        self.pre_ctx = torch.empty(0)
        self.pre_ctx_v = torch.empty(0)

        self.mlp_512 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(ctx_dim, ctx_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(ctx_dim * 4, ctx_dim))
        ]))
        self.mlp_768 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(768, 768 * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(768 * 4, 768))
        ]))
        self.ln_2_512 = LayerNorm(ctx_dim)
        self.ln_2_768 = LayerNorm(768)

    def forward(self, ctx, ctx_v):
        ctx.to(self.dtype)
        ctx_v.to(self.dtype)

        with torch.no_grad():
            temp_pre_ctx = self.pre_ctx
            temp_pre_ctx_v = self.pre_ctx_v

        ctx_t2i, attention_weights0 = self.multi_t2i(ctx, temp_pre_ctx)  # [n_ctx, 768]
        ctx_i2t, attention_weights1 = self.multi_i2t(ctx_v, temp_pre_ctx_v)  # [1, n_ctx, 512]

        ctx_t2i = ctx_t2i + self.mlp_768(ctx_t2i.to(torch.float32)).to(self.dtype)
        ctx_i2t = ctx_i2t + self.mlp_512(ctx_i2t.to(torch.float32)).to(self.dtype)

        self.pre_ctx = ctx_t2i
        self.pre_ctx_v = ctx_i2t

        return ctx_i2t, ctx_t2i


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # concat shared_ctx # [500, 2*n_ctx, 512] with x
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class ProgressiveCoPromptingLearning(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROGCOPL.N_CTX  # 每一侧prompt个数
        ctx_init = cfg.TRAINER.PROGCOPL.CTX_INIT  # 第一层输入的提示词，“a photo of”
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        self.ctx_dim = ctx_dim
        clip_imsize = clip_model.visual.input_resolution  # 224
        cfg_imsize = cfg.INPUT.SIZE[0]  # 224
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.PROGCOPL.PROMPT_DEPTH >= 1, "For ProgCoPL, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.PROGCOPL.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 文本->图像 引导 ctx: 文本侧的prompt， self.proj_to_v(ctx): 文本引导下的视觉prompt
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]  # [0, 0, :] 是SOS对应的embedding，要去掉
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('ProgCoPL design: Evo-modal Co-Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of ProgCoPL context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        # self.proj_t2i = nn.Linear(ctx_dim, 768)  # 文本侧的prompt都是512维的，图像侧是768维的，需要一个映射
        # self.proj_t2i.half()
        # self.proj_i2t = nn.Linear(768, ctx_dim)
        # self.proj_i2t.half()

        self.m_prompt_generator = MutualPromptGeneration(ctx_dim, dtype)

        self.ctx = nn.Parameter(ctx_vectors)  # 将required_graid转为True, 从而在训练过程中可以更新这些ctx token
        # 图像->文本 引导 ctx_v: 图像侧的prompt， self.proj_to_t(ctx_v): 视觉引导下的文本prompt
        ctx_vectors_v = torch.empty(n_ctx, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors_v, std=0.02)
        self.ctx_v = nn.Parameter(ctx_vectors_v)

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow ProgCoPL
        # compound prompts
        # text->image
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt #compound_prompt_projections就是论文中的F
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        # image->text
        self.compound_prompts_image = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                        for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_image:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(768, ctx_dim)
        self.compound_prompt_projections_i2t = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + 2 * n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx_t = self.ctx  # 文本侧第一层输入的红色P0 [n_ctx, 512]
        ctx_v = self.ctx_v  # 图像侧第一层添加的红色prompt [n_ctx, 768]

        if ctx_t.dim() == 2:
            ctx_t = ctx_t.unsqueeze(0).expand(self.n_cls, -1, -1)  # [500, n_ctx, 512]
        if ctx_v.dim() == 2:
            ctx_v = ctx_v.unsqueeze(0)  # [1, n_ctx, 768]

        ctx_i2t, ctx_t2i = self.m_prompt_generator(ctx_t, ctx_v)

        # ctx_i2t = self.proj_i2t(ctx_v)  # [1, n_ctx, 512]
        ctx_i2t = ctx_i2t.expand(self.n_cls, -1, -1)  # [500, n_ctx, 512]
        ctx_t_l0 = torch.cat([ctx_t, ctx_i2t], dim=1)  # [500, 2*n_ctx, 512]

        # ctx_t2i = self.proj_t2i(self.ctx)  # [n_ctx, 768]
        ctx_v_l0 = torch.cat([self.ctx_v, ctx_t2i[0]], dim=0)  # [2*n_ctx, 768]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx_t_l0, prefix, suffix)  # 文本侧第一层输入的 [SOS P0 w1, w2,....] w蕴含在token_suffix中了

        # 初始化1-prompt-depths层图像侧和文本侧prompt，包括自己添加的prompt和文本引导来的prompt
        visual_deep_prompts = []
        textual_deep_prompts = []
        for index in range(len(self.compound_prompt_projections)):
            visual_prompt = self.compound_prompts_image[index]  # [n_ctx, 768]
            textual_prompt = self.compound_prompts_text[index]  # [n_ctx, 512]
            t2i_prompt_gen_layer = self.compound_prompt_projections[index]
            i2t_prompt_gen_layer = self.compound_prompt_projections_i2t[index]
            t2i_prompt = t2i_prompt_gen_layer(textual_prompt)  # [n_ctx, 768]
            i2t_prompt = i2t_prompt_gen_layer(visual_prompt)  # [n_ctx, 512]
            v_prompt = torch.cat([visual_prompt, t2i_prompt], dim=0)  # [2*n_ctx, 768]
            t_prompt = torch.cat([textual_prompt, i2t_prompt], dim=0)  # [2*n_ctx, 512]
            visual_deep_prompts.append(v_prompt)
            textual_deep_prompts.append(t_prompt)

        # prompts: 文本侧第一层输入的完整序列，ctx_t_l0:文本侧第一层添加的prompt， ctx_v_l0:图像侧第一层添加的prompt
        # visual_deep_prompts: 1-J层图像侧添加的prompt，textual_deep_prompts：文本侧1-J层添加的prompt
        return prompts, ctx_v_l0, textual_deep_prompts, visual_deep_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = ProgressiveCoPromptingLearning(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx_v_l0, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        # prompts: 第一层加入的P0, tokenized_prompts: a photo of <class name>对应的token，
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx_v_l0, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class ProgCoPL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROGCOPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROGCOPL.PREC == "fp32" or cfg.TRAINER.PROGCOPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalCoPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROGCOPL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROGCOPL.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward(retain_graph=True)
            # torch.autograd.set_detect_anomaly(True)
            # loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"模型的训练参数总数: {total_params}")

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)