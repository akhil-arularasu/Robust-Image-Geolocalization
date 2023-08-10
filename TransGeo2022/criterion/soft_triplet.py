import torch
from torch import nn
import numpy as np

# this is equivalent to the loss function in CVMNet with alpha=10, here we simplify it with cosine similarity
class SoftTripletBiLoss(nn.Module):
    def __init__(self, margin=None, alpha=20, **kwargs):
        super(SoftTripletBiLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, inputs_q_clean, inputs_q_noisy, inputs_k):
        loss_1_clean, mean_pos_sim_1_clean, mean_neg_sim_1_clean = self.single_forward(inputs_q_clean, inputs_k)
        loss_2_clean, mean_pos_sim_2_clean, mean_neg_sim_2_clean = self.single_forward(inputs_k, inputs_q_clean)
        
        loss_clean = (loss_1_clean + loss_2_clean) / 2
        mean_pos_sim_clean = (mean_pos_sim_1_clean + mean_pos_sim_2_clean) / 2
        mean_neg_sim_clean = (mean_neg_sim_1_clean + mean_neg_sim_2_clean) / 2

        loss_1_noisy, mean_pos_sim_1_noisy, mean_neg_sim_1_noisy = self.single_forward(inputs_q_noisy, inputs_k)
        loss_2_noisy, mean_pos_sim_2_noisy, mean_neg_sim_2_noisy = self.single_forward(inputs_k, inputs_q_noisy)
    
        loss_noisy = (loss_1_noisy + loss_2_noisy) / 2
        mean_pos_sim_noisy = (mean_pos_sim_1_noisy + mean_pos_sim_2_noisy) / 2
        mean_neg_sim_noisy = (mean_neg_sim_1_noisy + mean_neg_sim_2_noisy) / 2
    
        return (loss_clean+loss_noisy)*0.5, (mean_pos_sim_clean+mean_pos_sim_noisy)*0.5, (mean_neg_sim_clean+mean_neg_sim_noisy)*0.5

    def single_forward(self, inputs_q, inputs_k):
        n = inputs_q.size(0)

        normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
        # Compute similarity matrix
        sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())

        # split the positive and negative pairs
        eyes_ = torch.eye(n).cuda()

        pos_mask = eyes_.eq(1)
        neg_mask = ~pos_mask

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        pos_sim_ = pos_sim.unsqueeze(dim=1).expand(n, n-1)
        neg_sim_ = neg_sim.reshape(n, n-1)

        loss_batch = torch.log(1 + torch.exp((neg_sim_ - pos_sim_) * self.alpha))
        if torch.isnan(loss_batch).any():
            print(inputs_q, inputs_k)
            raise Exception

        loss = loss_batch.mean()

        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()
        return loss, mean_pos_sim, mean_neg_sim
