import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')

# --------------------
# DataLoader Options
# --------------------

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--data_dir', type=str, default='/isize2/sain/data/Sketchy/') 
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)
parser.add_argument(
    '--retrieval_level',
    type=str,
    default='category',
    choices=['category', 'fine_grain'],
    help='Use category-level SBIR labels or fine-grained sketch-photo instance labels.')

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-5)
parser.add_argument('--prompt_lr', type=float, default=1e-5)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--max_epochs', type=int, default=60)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--lambda_cls', type=float, default=0.5)
parser.add_argument('--lambda_divergence', type=float, default=0.1)
parser.add_argument('--lambda_patch_shuffle', type=float, default=1.0)
parser.add_argument('--patch_grid', type=int, default=2)
parser.add_argument(
    '--logger',
    type=str,
    default='csv',
    choices=['csv', 'tensorboard', 'none'],
    help='Logger backend. CSV avoids TensorBoard/TensorFlow dependency issues.')

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

opts = parser.parse_args()
