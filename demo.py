import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.getcwd())

import os.path as osp
import pickle
from functools import lru_cache

import cv2
import imageio
import numpy as np
import pyrender
import smplx
import torch
import trimesh
from models.consts import speaker_id
from moviepy.editor import AudioFileClip, VideoFileClip
from pyrender.constants import RenderFlags
from tqdm import tqdm
from trainer.config import load_JsonConfig
from trainer.options import parse_args
from trainer.smplx_body_retnet import TrainWrapper as s2a_body_retnet
from trainer.smplx_body_vq import TrainWrapper as s2a_body_vq
from trainer.smplx_face import TrainWrapper as s2a_face
from transformers import Wav2Vec2Processor


@lru_cache(maxsize=1)
def _get_smplx_model():
    # Cache the SMPL-X model to avoid re-loading weights repeatedly (can be very memory-heavy)
    return smplx.create(
        model_path=r"visualise/smplx/SMPLX_NEUTRAL.npz",
        model_type="smplx",
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext="npz",
        use_pca=False,
        dtype=torch.float32,
    )


def render_pose(beta, pose, expression, file_name, audio_path):
    model = _get_smplx_model()
    if torch.cuda.is_available():
        pose = pose.cuda()
        model = model.cuda()
        betas = torch.tensor(beta).to(torch.float32).cuda()
    else:
        pose = pose.cpu()
        model = model.cpu()
        betas = torch.tensor(beta).to(torch.float32)

    jaw_pose = expression[:, 0:3]  # (1*3)
    leye_pose = pose[:, 6:9]  # (1*3)
    reye_pose = pose[:, 9:12]  # (1*3)
    global_orient = pose[:, 12:15]  # (1*3)
    body_pose = pose[:, 15:78]  # (21*3)
    trans_pose = pose[:, 0:3]  # (1*3)
    left_hand_pose = pose[:, 78:123]  # (15*3)
    right_hand_pose = pose[:, 123:168]  # (15*3)
    expression = expression[:, 3:]
    with torch.no_grad():
        output = model(
            betas=betas,
            expression=expression,
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=trans_pose,
            return_verts=True,
        )

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    vertices_all = np.array(vertices).reshape(-1, 10475, 3)
    z_min = vertices_all[0][0][1]

    for i in vertices_all:
        min_z = min(i[:, 1])
        z_min = min(z_min, min(i[:, 1]))

    x_min = -1.5
    x_max = +1.5
    y_min = -3
    y_max = +3

    ground = np.array(
        [
            [x_min, z_min, y_min],
            [x_min, z_min, y_max],
            [x_max, z_min, y_min],
            [x_max, z_min, y_max],
        ]
    ).reshape(-1, 3)

    # Stream frames directly to dist to avoid holding the whole video in RAM.
    os.makedirs("videos", exist_ok=True)
    os.makedirs("test_pictures", exist_ok=True)
    video_path = osp.join(r"videos", file_name + ".mp4")
    writer = imageio.get_writer(video_path, fps=30)
    renderer = None

    for frame, vertices in tqdm(
        enumerate(vertices_all), total=len(vertices_all), desc="Rendering", unit="frame"
    ):
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(np.dot(1.8, vertices), model.faces, process=False)
        color_1 = [0.71, 0.51, 0.27]  ##BGR
        material_1 = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(235 / 255, 180 / 255, 163 / 255, 1.0),
        )
        mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material_1)
        scene = pyrender.Scene(
            bg_color=[0.15, 0.15, 0.15, 1.0], ambient_light=(0.3, 0.3, 0.3)
        )
        scene.add(mesh, "mesh")

        gro_mesh = trimesh.Trimesh(
            np.dot(1.8, ground), [[0, 1, 2], [3, 2, 1]], process=False
        )

        color_2 = [0.07, 0.27, 0.8]  ##BGR
        material_2 = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5, alphaMode="OPAQUE", baseColorFactor=(0.2, 0.2, 0.2, 1.0)
        )
        gro_mesh = pyrender.Mesh.from_trimesh(gro_mesh, material=material_2)
        scene.add(gro_mesh, "gro_mesh")

        light_pose = np.eye(4)

        spot_l = pyrender.SpotLight(
            color=np.ones(3),
            intensity=10.0,
            innerConeAngle=np.pi / 2,
            outerConeAngle=np.pi / 2,
        )
        light_pose[:3, 3] = [2, 2, 2]
        scene.add(spot_l, pose=light_pose)
        light_pose[:3, 3] = [-2, 2, 2]
        scene.add(spot_l, pose=light_pose)

        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
            pose=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.8, 0.6, 0.0],
                    [0.0, -0.6, 0.8, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

        focal_length = [5000, 5000]
        img = np.zeros((1000, 1000, 3))
        scale_ratio = 1
        resolution = np.array(img.shape[:2]) * scale_ratio

        cam = np.array([1, 0, 0])
        cam = cam.copy()
        sx, tx, ty = cam
        sy = sx

        camera_translation = np.array(
            [-tx, ty, 2 * focal_length[0] / (resolution[0] * sy + 1e-9)]
        )

        render_res = resolution
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length[0] / 1.2,
            fy=focal_length[1] / 1.2,
            cx=render_res[1],
            cy=render_res[0] / 1.4,
        )

        camera_rotation = np.eye(3)
        camera_rotation = camera_rotation.T
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = camera_rotation @ camera_translation
        scene.add(camera, pose=camera_pose)

        if renderer is None:
            renderer = pyrender.OffscreenRenderer(
                viewport_width=resolution[0] * 2,
                viewport_height=resolution[1] * 2,
                point_size=1.0,
            )
        rgb, _ = renderer.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL)
        # OpenCV uses BGR order by default
        cv2.imwrite(
            osp.join(r"test_pictures", str(frame) + ".jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        # imageio expects RGB order, pyrender returns RGB already, keep as-is.
        writer.append_data(rgb)

        # Help GC for large per-frame objects (scene contains GPU handles/textures/etc.)
        try:
            scene.clear()
        except Exception:
            pass
        del scene

    # Ensure we release renderer + writer even if rendering errors occur.
    writer.close()
    if renderer is not None:
        try:
            renderer.delete()
        except Exception:
            pass

    # Add audio track and close moviepy resources explicitly
    video = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    # Ensure audio duration matches the generated video duration to avoid empty/trimmed audio tracks
    try:
        audio_clip = audio_clip.set_duration(video.duration)
    except Exception:
        pass
    videos = video.set_audio(audio_clip)
    try:
        videos.write_videofile(
            osp.join(r"videos", file_name + "_audio.mp4"),
            codec="libx264",
            fps=30,
            audio_codec="aac",
            temp_audiofile=osp.join(r"videos", file_name + "_temp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None,
        )
    finally:
        try:
            videos.close()
        except Exception:
            pass
        try:
            video.close()
        except Exception:
            pass
        try:
            audio_clip.close()
        except Exception:
            pass
        # Keep the silent video file (`video_path`) by default.
        # If you want to auto-clean it up, uncomment the block below.
        # try:
        #     os.remove(video_path)
        # except Exception:
        #     pass


def init_model(model_name, model_path, args, config):
    if model_name == "s2a_face":
        generator = s2a_face(
            args,
            config,
        )
    elif model_name == "s2a_body_vq":
        generator = s2a_body_vq(
            args,
            config,
        )
    elif model_name == "s2a_body_retnet":
        generator = s2a_body_retnet(
            args,
            config,
        )
    else:
        raise NotImplementedError

    model_ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    if model_name == "smplx_S2G":
        generator.generator.load_state_dict(model_ckpt["generator"]["generator"])

    elif "generator" in list(model_ckpt.keys()):
        generator.load_state_dict(model_ckpt["generator"])
    else:
        model_ckpt = {"generator": model_ckpt}
        generator.load_state_dict(model_ckpt)

    return generator


def infer(g_body, g_face, config, args):
    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000

    cur_wav_file = args.audio_file
    cur_text_file = args.text_file
    face = args.only_face

    print("input audio:", cur_wav_file)
    for speaker in args.speaker_names:
        # id = torch.tensor(speaker_id[speaker]).cuda()

        f = open(r"config/betas.pkl", "rb+")
        speaker_beta = pickle.load(f)
        beta = speaker_beta[speaker]

        id = torch.tensor([speaker_id[speaker]], device=args.device)

        pred_face = g_face.infer_on_audio(cur_wav_file, id=id, am=am, am_sr=am_sr)
        pred_face = torch.tensor(pred_face).squeeze().to(args.device)

        pred = g_body.infer_on_audio(cur_wav_file, id=id, fps=30).squeeze(dim=0)
        pred = torch.tensor(pred).squeeze().to(args.device)
        if pred.shape[0] < pred_face.shape[0]:
            repeat_frame = (
                pred[-1].unsqueeze(dim=0).repeat(pred_face.shape[0] - pred.shape[0], 1)
            )
            pred = torch.cat([pred, repeat_frame], dim=0)
        else:
            pred = pred[: pred_face.shape[0], :]
        save_pred = torch.cat(
            [pred[:, :3], pred_face[:, :3], pred[:, 6:], pred_face[:, 3:]], dim=1
        )
        print(speaker, "-", "pose:")
        render_pose(
            beta,
            pred,
            pred_face,
            cur_wav_file.split("/")[-1].split(".")[0] + "_" + speaker,
            cur_wav_file,
        )


def main():
    parser = parse_args()
    args = parser.parse_args()
    # device = torch.device(args.gpu)
    # torch.cuda.set_device(device)

    if torch.cuda.is_available():
        args.device = "cuda"
    # elif torch.backends.mps.is_available():
    #    args.device = "mps"
    else:
        args.device = "cpu"

    config = load_JsonConfig(args.config_file)

    face_model_name = args.face_model_name
    face_model_path = args.face_model_path
    body_model_name = args.body_model_name
    body_model_path = args.body_model_path

    generator = init_model(body_model_name, body_model_path, args, config)
    generator_face = init_model(face_model_name, face_model_path, args, config)

    infer(generator, generator_face, config, args)


if __name__ == "__main__":
    main()
