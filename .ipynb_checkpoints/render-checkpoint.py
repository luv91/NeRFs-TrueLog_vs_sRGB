import glob
import logging
import os
import sys
import time

from absl import app
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import checkpoints
from internal import utils
from internal import vis
from matplotlib import cm
import mediapy as media
import torch
import numpy as np
import accelerate
import imageio
from torch.utils._pytree import tree_map


def render_frame_with_camera_position(config, pos):
    """
    Placeholder function for a 360° rotation.
    In practice, you'd run a forward pass using `models.render_image(...)`
    with the camera extrinsics corresponding to 'pos'. 
    Return a NumPy image in [0..255] or [0..1] for appending to video.
    """
    return None


def find_mse_folder_if_missing(out_dir, idx_str, base_dir, zpad):
    """
    If 'mse_map_000.tiff' is not found in out_dir, this function tries to
    locate it by scanning subdirectories like 'test_preds_step_XXX' or
    'path_renders_step_XXX'. Returns the corrected folder if found.
    """
    mse_zerofile = os.path.join(out_dir, f'mse_map_{idx_str}.tiff')
    if utils.file_exists(mse_zerofile):
        return out_dir  # It's already here.

    # Otherwise, scan for any subfolder containing 'mse_map_000.tiff'.
    # We'll just look under 'base_dir' or immediate subfolders for demonstration.
    print(f"No 'mse_map_{idx_str}.tiff' found in '{out_dir}'. Trying to find MSE maps elsewhere...")
    candidates = glob.glob(os.path.join(base_dir, '*_step_*'))
    for cand in candidates:
        testfile = os.path.join(cand, f'mse_map_{idx_str}.tiff')
        if utils.file_exists(testfile):
            print(f"Found MSE maps in '{cand}' instead.")
            return cand

    print("Could not locate a fallback folder with 'mse_map_000.tiff'.")
    return out_dir  # fallback to original anyway


def create_videos(config, base_dir, out_dir, out_name, num_frames):
    """Creates videos out of the images saved to disk."""
    names = [n for n in config.exp_path.split(os.sep) if n]
    # Last two parts of checkpoint path are experiment name and scene name.
    exp_name, scene_name = names[-2:]
    video_prefix = f'{scene_name}_{exp_name}_{out_name}'

    zpad = max(3, len(str(num_frames - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)

    utils.makedirs(base_dir)

    # Detect shape / dynamic range for distance maps
    depth_file = os.path.join(out_dir, f'distance_mean_{idx_to_str(0)}.tiff')
    img_file = os.path.join(out_dir, f'color_{idx_to_str(0)}.png')

    if utils.file_exists(depth_file):
        depth_frame = utils.load_img(depth_file)
        shape = depth_frame.shape
        p = config.render_dist_percentile
        distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
        depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
        lo, hi = distance_limits
    else:
        if utils.file_exists(img_file):
            img_frame = utils.load_img(img_file)
            shape = img_frame.shape
        else:
            shape = [1, 1]  # fallback
        lo, hi = 0.0, 1.0
        depth_curve_fn = None

    print(f'Video shape is {shape[:2]}')

    # ----------------------------------------------------------------------
    # Try to find the MSE maps in out_dir or a fallback folder
    # ----------------------------------------------------------------------
    out_dir_corrected = find_mse_folder_if_missing(out_dir, idx_to_str(0), base_dir, zpad)

    # --- Create MSE video if mse_map is present in out_dir_corrected ---
    mse_zerofile = os.path.join(out_dir_corrected, f'mse_map_{idx_to_str(0)}.tiff')
    if utils.file_exists(mse_zerofile):
        video_file = os.path.join(base_dir, f'{video_prefix}_mse_map.mp4')
        print(f'Making MSE video: {video_file}...')
        writer = imageio.get_writer(video_file, fps=config.render_video_fps)

        for idx in range(num_frames):
            mse_map_file = os.path.join(out_dir_corrected, f'mse_map_{idx_to_str(idx)}.tiff')
            if not utils.file_exists(mse_map_file):
                raise ValueError(f'Expected MSE map file not found: {mse_map_file}')

            # Load MSE map [H, W], single channel
            img = utils.load_img(mse_map_file)
            # Apply colormap
            img = vis.visualize_cmap(img, np.ones_like(img), cm.get_cmap('turbo'))
            frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
            writer.append_data(frame)

        writer.close()
        print(f'MSE video saved to: {video_file}')
    else:
        print(f"No 'mse_map_000.tiff' found in '{out_dir_corrected}', skipping MSE video generation.")

    # ----------------------------------------------------------------------
    # Create videos for color, normals, acc, distance, depth_triplet
    # ----------------------------------------------------------------------
    for k in ['color', 'normals', 'acc', 'distance_mean', 'distance_median', 'depth_triplet']:
        video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
        file_ext = 'png' if k in ['color', 'normals'] else 'tiff'
        file0 = os.path.join(out_dir, f'{k}_{idx_to_str(0)}.{file_ext}')

        if not utils.file_exists(file0):
            print(f"Images missing for tag '{k}' in '{out_dir}', skipping.")
            continue

        print(f'Making video {video_file}...')
        writer = imageio.get_writer(video_file, fps=config.render_video_fps)

        for idx in range(num_frames):
            img_file = os.path.join(out_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
            if not utils.file_exists(img_file):
                raise ValueError(f'Image file {img_file} does not exist.')

            img = utils.load_img(img_file)
            if k in ['color', 'normals']:
                img = img / 255.
            elif k.startswith('distance') or k == 'depth_triplet':
                if depth_curve_fn is not None:
                    img = vis.visualize_cmap(img, np.ones_like(img), cm.get_cmap('turbo'), lo, hi, curve_fn=depth_curve_fn)
                else:
                    img = vis.visualize_cmap(img, np.ones_like(img), cm.get_cmap('turbo'))

            frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
            writer.append_data(frame)

        writer.close()

    # ----------------------------------------------------------------------
    # Example 360° rotation video placeholder
    # ----------------------------------------------------------------------
    print("Generating 360° rotation video for one scene...")

    video_file_360 = os.path.join(base_dir, f'{video_prefix}_360_rotation.mp4')
    camera_positions = []
    for i in range(360):
        theta = np.radians(i)
        x = np.cos(theta)
        y = np.sin(theta)
        z = 0
        camera_positions.append((x, y, z))

    print(f'Creating 360° rotation video: {video_file_360}')
    writer = imageio.get_writer(video_file_360, fps=60)  # 6 seconds at 60 fps

    for idx, pos in enumerate(camera_positions):
        print(f'Rendering frame {idx + 1}/360 at {pos}')
        frame = render_frame_with_camera_position(config, pos)  # Implement your logic
        if frame is not None:
            writer.append_data(frame)

    writer.close()
    print(f"360° rotation video saved at: {video_file_360}")


def main(unused_argv):
    config = configs.load_config()
    config.exp_path = os.path.join('exp', config.exp_name)
    config.checkpoint_dir = os.path.join(
        config.exp_path, 
        'checkpoints' if not config.render_ft else f'ft/{config.ft_name}/checkpoints/'
    )
    config.render_dir = os.path.join(
        config.exp_path,
        'render' if not config.render_ft else f'ft/{config.ft_name}/render'
    )

    accelerator = accelerate.Accelerator()
    logging.basicConfig(
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(
                    config.exp_path,
                    'log_render.txt' if not config.render_ft else f'ft/{config.ft_name}/log_render.txt'
                )
            )
        ],
        level=logging.INFO,
    )
    sys.excepthook = utils.handle_exception
    logger = accelerate.logging.get_logger(__name__)
    logger.info(config)
    logger.info(accelerator.state, main_process_only=False)

    config.world_size = accelerator.num_processes
    config.global_rank = accelerator.process_index
    accelerate.utils.set_seed(config.seed, device_specific=True)

    dataset = datasets.load_dataset('test', config.data_dir, config)
    dataloader = torch.utils.data.DataLoader(
        np.arange(len(dataset)),
        shuffle=False,
        batch_size=1,
        collate_fn=dataset.collate_fn,
    )
    dataiter = iter(dataloader)
    if config.rawnerf_mode:
        postprocess_fn = dataset.metadata['postprocess_fn']
    else:
        postprocess_fn = lambda z: z

    dataset_info_for_model = {'size': dataset.size}
    model = models.Model(config=config, dataset_info=dataset_info_for_model)
    model.eval()
    model = accelerator.prepare(model)
    step = checkpoints.restore_checkpoint(
        config.checkpoint_dir, accelerator, logger, strict=config.load_state_dict_strict
    )
    logger.info(f'Rendering checkpoint at step {step}.')

    # Decide the subfolder name for outputs
    if config.render_path:
        out_name = 'path_renders'
    elif config.render_train:
        out_name = 'train_preds'
    else:
        out_name = 'test_preds'
    out_name = f'{out_name}_step_{step}'

    out_dir = os.path.join(config.render_dir, out_name)
    utils.makedirs(out_dir)

    path_fn = lambda x: os.path.join(out_dir, x)

    # Zero-padding for filenames
    zpad = max(3, len(str(dataset.size - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)

    for idx in range(dataset.size):
        batch = next(dataiter)
        idx_str = idx_to_str(idx)
        curr_file = path_fn(f'color_{idx_str}.png')

        if utils.file_exists(curr_file):
            logger.info(f'Image {idx + 1}/{dataset.size} already exists, skipping')
            continue

        batch = tree_map(lambda x: x.to(accelerator.device) if x is not None else None, batch)
        logger.info(f'Evaluating image {idx + 1}/{dataset.size}')

        eval_start_time = time.time()
        rendering = models.render_image(model, accelerator, batch, False, 1, config)
        logger.info(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

        if accelerator.is_main_process:
            rendering = tree_map(lambda x: x.detach().cpu().numpy() if x is not None else None, rendering)
            rendering['rgb'] = postprocess_fn(rendering['rgb'])
            utils.save_img_u8(rendering['rgb'], path_fn(f'color_{idx_str}.png'))

            if 'normals' in rendering:
                utils.save_img_u8(rendering['normals'] / 2. + 0.5, path_fn(f'normals_{idx_str}.png'))

            if not config.render_ft:
                utils.save_img_f32(rendering['distance_mean'], path_fn(f'distance_mean_{idx_str}.tiff'))
                utils.save_img_f32(rendering['distance_median'], path_fn(f'distance_median_{idx_str}.tiff'))

                if 'depth_triplet' in rendering:
                    utils.save_img_f32(rendering['depth_triplet'], path_fn(f'depth_triplet_{idx_str}.tiff'))

                if config.save_acc_maps:
                    utils.save_img_f32(rendering['acc'], path_fn(f'acc_{idx_str}.tiff'))

    # Once all images are saved, create videos
    num_files = len(glob.glob(path_fn('color_*.png')))
    if accelerator.is_main_process and num_files == dataset.size:
        logger.info('All files found, creating videos.')
        create_videos(config, config.render_dir, out_dir, out_name, dataset.size)

    accelerator.wait_for_everyone()
    logger.info('Finish rendering.')


if __name__ == '__main__':
    configs.define_common_flags()
    with gin.config_scope('eval'):  # Use the same scope as eval.py
        app.run(main)
