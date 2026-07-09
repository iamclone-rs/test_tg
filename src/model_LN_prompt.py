import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)

        # Prompt Engineering
        if self.opts.retrieval_level == 'fine_grain':
            self.prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        else:
            self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
            self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=self.opts.margin)

        self.best_metric = -1e3
        self.validation_outputs = []

    def configure_optimizers(self):
        prompt_params = [self.prompt] if self.opts.retrieval_level == 'fine_grain' else [self.sk_prompt, self.img_prompt]
        optimizer = torch.optim.Adam([
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': prompt_params, 'lr': self.opts.prompt_lr}])
        return optimizer

    def forward(self, data, dtype='image'):
        if self.opts.retrieval_level == 'fine_grain':
            feat = self.clip.encode_image(
                data, self.prompt.expand(data.shape[0], -1, -1))
        elif dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def _classification_loss(self, sk_feat, img_feat, category):
        categories = list(category)
        unique_categories = sorted(set(categories))
        class_to_idx = {name: idx for idx, name in enumerate(unique_categories)}
        labels = torch.tensor([class_to_idx[name] for name in categories], device=sk_feat.device)
        prompts = ['a photo of a %s.' % name.replace('_', ' ') for name in unique_categories]
        text = clip.tokenize(prompts).to(sk_feat.device)
        text_feat = self.clip.encode_text(text)

        sk_norm = F.normalize(sk_feat.float(), dim=1)
        img_norm = F.normalize(img_feat.float(), dim=1)
        text_norm = F.normalize(text_feat.float(), dim=1)
        logit_scale = self.clip.logit_scale.exp().float()

        sk_logits = logit_scale * sk_norm @ text_norm.t()
        img_logits = logit_scale * img_norm @ text_norm.t()
        return F.cross_entropy(sk_logits, labels) + F.cross_entropy(img_logits, labels)

    def _divergence_loss(self, sk_feat, img_feat, neg_feat, category, bins=16):
        categories = list(category)
        deltas = self.distance_fn(sk_feat, neg_feat) - self.distance_fn(sk_feat, img_feat)
        unique_categories = sorted(set(categories))
        if len(unique_categories) < 2:
            return deltas.new_zeros(())

        centers = torch.linspace(-1.0, 1.0, bins, device=deltas.device, dtype=deltas.dtype)
        distributions = []
        for name in unique_categories:
            mask = torch.tensor([cat == name for cat in categories], device=deltas.device)
            if mask.sum() == 0:
                continue
            cat_delta = deltas[mask].unsqueeze(1)
            weights = torch.softmax(-((cat_delta - centers) ** 2) / 0.05, dim=1)
            hist = weights.mean(dim=0)
            distributions.append(hist / hist.sum().clamp_min(1e-6))

        if len(distributions) < 2:
            return deltas.new_zeros(())

        loss = deltas.new_zeros(())
        pairs = 0
        for i, first in enumerate(distributions):
            for j, second in enumerate(distributions):
                if i == j:
                    continue
                loss = loss + F.kl_div((first + 1e-6).log(), second + 1e-6, reduction='batchmean')
                pairs += 1
        return loss / max(pairs, 1)

    def _shuffle_patches(self, images, permutation):
        grid = self.opts.patch_grid
        if grid <= 1:
            return images
        batch, channels, height, width = images.shape
        if height % grid != 0 or width % grid != 0:
            raise RuntimeError('Image size must be divisible by patch_grid for patch shuffling.')

        patch_h, patch_w = height // grid, width // grid
        patches = images.reshape(batch, channels, grid, patch_h, grid, patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, grid * grid, channels, patch_h, patch_w)
        patches = patches[:, permutation]
        patches = patches.reshape(batch, grid, grid, channels, patch_h, patch_w)
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)

    def _patch_shuffle_loss(self, sk_tensor, img_tensor):
        grid = self.opts.patch_grid
        patch_count = grid * grid
        permutation_pos = torch.randperm(patch_count, device=sk_tensor.device)
        permutation_neg = torch.randperm(patch_count, device=sk_tensor.device)
        while torch.equal(permutation_pos, permutation_neg) and patch_count > 1:
            permutation_neg = torch.randperm(patch_count, device=sk_tensor.device)

        sk_shuffled = self._shuffle_patches(sk_tensor, permutation_pos)
        img_pos = self._shuffle_patches(img_tensor, permutation_pos)
        img_neg = self._shuffle_patches(img_tensor, permutation_neg)

        sk_feat = self.forward(sk_shuffled, dtype='sketch')
        img_pos_feat = self.forward(img_pos, dtype='image')
        img_neg_feat = self.forward(img_neg, dtype='image')
        pos_dist = self.distance_fn(sk_feat, img_pos_feat)
        neg_dist = self.distance_fn(sk_feat, img_neg_feat)
        return F.relu(self.opts.margin + pos_dist - neg_dist).mean()

    @staticmethod
    def _average_precision(scores, target):
        order = torch.argsort(scores, descending=True)
        sorted_target = target[order].float()
        positives = sorted_target.sum()
        if positives == 0:
            return scores.new_zeros(())
        ranks = torch.arange(1, len(scores) + 1, device=scores.device, dtype=scores.dtype)
        precision_at_k = torch.cumsum(sorted_target, dim=0) / ranks
        return (precision_at_k * sorted_target).sum() / positives

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        batch_size = sk_tensor.shape[0]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        if self.opts.retrieval_level == 'fine_grain':
            cls_loss = self._classification_loss(sk_feat, img_feat, category)
            div_loss = self._divergence_loss(sk_feat, img_feat, neg_feat, category)
            ps_loss = self._patch_shuffle_loss(sk_tensor, img_tensor)
            loss = (
                loss
                + self.opts.lambda_cls * cls_loss
                + self.opts.lambda_divergence * div_loss
                + self.opts.lambda_patch_shuffle * ps_loss
            )
            self.log('train_cls_loss', cls_loss, prog_bar=False, batch_size=batch_size)
            self.log('train_div_loss', div_loss, prog_bar=False, batch_size=batch_size)
            self.log('train_ps_loss', ps_loss, prog_bar=False, batch_size=batch_size)
        self.log('train_loss', loss, prog_bar=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        batch_size = sk_tensor.shape[0]
        target_id = batch[5]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss, prog_bar=False, batch_size=batch_size)
        output = {
            'sk_feat': sk_feat.detach(),
            'img_feat': img_feat.detach(),
            'category': category,
            'target_id': target_id,
        }
        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def on_validation_epoch_end(self):
        Len = len(self.validation_outputs)
        if Len == 0:
            return
        query_feat_all = torch.cat([self.validation_outputs[i]['sk_feat'] for i in range(Len)])
        gallery_feat_all = torch.cat([self.validation_outputs[i]['img_feat'] for i in range(Len)])
        all_category = np.array(sum([list(self.validation_outputs[i]['category']) for i in range(Len)], []))
        all_target_id = np.array(sum([list(self.validation_outputs[i]['target_id']) for i in range(Len)], []))


        ## mAP retrieval metric over the positive image gallery, following the original code path.
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            if self.opts.retrieval_level == 'fine_grain':
                target[np.where(all_target_id == all_target_id[idx])] = True
            else:
                target[np.where(all_category == all_category[idx])] = True
            ap[idx] = self._average_precision(distance.cpu(), target.cpu())
        
        mAP = torch.mean(ap)
        self.log('mAP', mAP, prog_bar=False, batch_size=len(query_feat_all))
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        print ('epoch={} {} mAP={:.4f} best_mAP={:.4f}'.format(
            self.current_epoch, self.opts.retrieval_level, mAP.item(), self.best_metric))
        self.validation_outputs = []
