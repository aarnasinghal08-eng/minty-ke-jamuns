import torch

def heteroscedastic_loss(pred_rgb, target_rgb, pred_var):
    """
    Physics-Aware loss weighing reconstruction errors by self-estimated uncertainty.
    
    Arguments:
        pred_rgb: Predicted RGB image tensor of shape (B, 3, H, W).
        target_rgb: Target ground truth RGB image tensor of shape (B, 3, H, W).
        pred_var: Predicted variance (uncertainty) tensor of shape (B, 1, H, W).
    """
    diff_sq = (pred_rgb - target_rgb) ** 2
    # Mean difference squared across color channels (R, G, B)
    mean_diff_sq = torch.mean(diff_sq, dim=1, keepdim=True) # B, 1, H, W
    
    # Loss term = (mean_diff_sq) / (2 * variance)
    loss_term = mean_diff_sq / (2 * pred_var)
    # Penalty term to prevent variance from growing to infinity
    reg_term = 0.5 * torch.log(pred_var)
    
    return torch.mean(loss_term + reg_term)
