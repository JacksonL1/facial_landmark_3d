'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda'
cfg.device_id = '0'
cfg.detect_type = "2D"

cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'model', 'deca_model.tar')
cfg.rect_model_path = os.path.join(cfg.deca_dir, 'model', 's3fd.pth')
cfg.landmark_model_path = os.path.join(cfg.deca_dir, 'model', '2DFAN4-11f355bf06.pth.tar')
cfg.checkpoint_fp = os.path.join(cfg.deca_dir, 'model', 'vdc_mobilenet1.tar')

# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(cfg.deca_dir, 'model', 'head_template.obj')
cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'model', 'generic_model.pkl')
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'model', 'landmark_embedding.npy')
cfg.model.tex_type = 'BFM'  # BFM, FLAME, albedoMM
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.thresh = 0.5
cfg.list_size = 3
cfg.resize_size = 1280
cfg.STD_SIZE = 120
cfg.model.use_tex = False
cfg.model.jaw_type = 'aa'  # default use axis angle, another option: euler

## details
cfg.model.n_detail = 128
cfg.model.max_z = 0.01
cfg.scale = 1.25

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.batch_size = 24
cfg.dataset.num_workers = 2
cfg.dataset.image_size = 224


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
