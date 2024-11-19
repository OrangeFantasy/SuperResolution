import os
from typing import List
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import json
import multiprocessing
from multiprocessing import Process
from tqdm import tqdm

CPU_COUNT = multiprocessing.cpu_count()
SPECULAR_VALID = bool(True)


def remove_files(root: str, label: str):
    for file in os.listdir(root):
        if label in file:
            print(f"Removing {file}")
            os.remove(os.path.join(root, file))


def check_dir(path: str) -> str:
    if not os.path.exists(path): 
        os.makedirs(path)
    return path


def save_as_fp16(name: str, array: np.ndarray, ext: str = ".npz"):
    array = array.astype(np.float16)
    if ext == ".npy":
        np.save(name + ext, array)
    elif ext == ".npz":
        np.savez_compressed(name + ext, array)
    else:
        raise TypeError("Unsupported file extension")


def load_exr_with_rgb_chw(path: str) -> np.ndarray:
    try:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, :3]
    except Exception as ex:
        print(f"Cannot load {path}")
        raise ex
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    return image


def load_motion_vector(path: str):
    # NOTE: Range from [0, 1] to [-h/2,+w/2, +h/2,+w/2]

    # R - dx, G - dy
    velocity = load_exr_with_rgb_chw(path)[:2]
    velocity = velocity - 0.5
    velocity[0] = -velocity[0]

    # Normalized distance converted to pixel distance.
    h, w = velocity.shape[-2:]
    velocity[0] = velocity[0] * (w / 2)
    velocity[1] = velocity[1] * (h / 2)

    return velocity


def merge_gbuffers(dirname: str, basename: str, material_types: list[str], metarials_channles: list[int]):
    # NOTE: Range: [0, 1], include normal material.
    assert len(material_types) == len(metarials_channles), "The length of material_types and metarials_channles must be the same."

    gbuffers = []
    for material, channel in zip(material_types, metarials_channles):
        buffer = load_exr_with_rgb_chw(os.path.join(dirname, basename + material + ".exr"))[:channel]
        
        # NOTE: Some specular materials are invalid.
        # if material == "Specular" and not SPECULAR_VALID:
            # buffer = np.zeros_like(buffer)
        
        gbuffers.append(buffer)

    return np.concatenate(gbuffers, axis=0)


def is_valid_frame(frame_id: int, shot_frames: list[list[int]]):
    for start, end in shot_frames:
        if frame_id != start and frame_id != end and start < frame_id < end:
            return True
    return False


def get_valid_frame_names(dirname: str, materials: str | list[str], shot_frames: list[list[int]]) -> list[str]:
    if isinstance(materials, str):
        materials = [materials]

    file_list = sorted(os.listdir(dirname))

    frame_names_without_metarial: list[str] = []
    for name in file_list:
        for material in materials:
            name = name.replace(material, "")
        frame_names_without_metarial.append(os.path.splitext(name)[0])

    frame_names = []
    for name in sorted(list(set(frame_names_without_metarial))):
        frame_id = int(name.split(".")[1])
        if is_valid_frame(frame_id, shot_frames):
            frame_names.append(name)

    return frame_names


def preprocess(src_root, tgt_root, shot_frames, config):
    frame_metarials = config["frame_metarial"]
    motion_vector_material = config["motion_vector_metarial"]
    gbuffer_materials = config["gbuffer_metarials"]
    gbuffer_material_channles = config["gbuffer_metarial_channels"]

    for dirname in os.listdir(src_root):
        if dirname == "MSAA_64x" or dirname == "MSAA_16x":
            continue
            # Get frame names without materials.
            frame_names = get_valid_frame_names(os.path.join(src_root, dirname), frame_metarials, shot_frames)
            
            # The Frame image rendered by the "PreTonemapHDRColor" material.
            tgt_dir = check_dir(os.path.join(tgt_root, "Reference"))
            for name in tqdm(frame_names, desc="Reference"):
                path = os.path.join(src_root, dirname, name + frame_metarials + ".exr")
                frame = load_exr_with_rgb_chw(path)
                save_as_fp16(os.path.join(tgt_dir, name), frame)
            
        elif dirname == "NoAA":
            continue
            # Get frame names without materials.
            materials = gbuffer_materials + [frame_metarials, motion_vector_material]
            frame_names = get_valid_frame_names(os.path.join(src_root, dirname), materials, shot_frames)

            # Reads and merges all gbuffers.
            tgt_dir = check_dir(os.path.join(tgt_root, "HR_GBuffers"))
            def _sub_process(_frame_names):
                for name in tqdm(_frame_names, desc="HR_GBuffers", ncols=150, nrows=24, position=multiprocessing.current_process()._identity[0]):
                    gbuffers = merge_gbuffers(os.path.join(src_root, dirname), name, gbuffer_materials, gbuffer_material_channles)
                    save_as_fp16(os.path.join(tgt_dir, name), gbuffers)
            
            tasks: list[Process] = []
            for i in range(CPU_COUNT):
                _frame_names = frame_names[i::CPU_COUNT]
                tasks.append(Process(target=_sub_process, args=(_frame_names, )))
            
            for task in tasks: task.start()
            for task in tasks: task.join()

            # for name in tqdm(frame_names, desc="HR_GBuffers"):
            #     gbuffers = merge_gbuffers(os.path.join(src_root, dirname), name, gbuffer_materials, gbuffer_material_channles)
            #     save_as_fp16(os.path.join(tgt_dir, name), gbuffers)

        elif dirname == "LR_4x" or dirname == "LRx4":
            continue    
            # Get frame names without materials.
            materials = gbuffer_materials + [frame_metarials, motion_vector_material]
            frame_names = get_valid_frame_names(os.path.join(src_root, dirname), materials, shot_frames)
            
            # Reads and merges all gbuffers.
            tgt_dir = check_dir(os.path.join(tgt_root, "LR_4x_GBuffers"))
            for name in tqdm(frame_names, desc="LR_4x_GBuffers"):
                # break
                gbuffers = merge_gbuffers(os.path.join(src_root, dirname), name, gbuffer_materials, gbuffer_material_channles)
                save_as_fp16(os.path.join(tgt_dir, name), gbuffers)
                
            # Reads and save motion vector.
            tgt_dir = check_dir(os.path.join(tgt_root, "LR_4x_MotionVector"))
            for name in tqdm(frame_names, desc="LR_4x_MotionVector"):
                # break
                path = os.path.join(src_root, dirname, name + motion_vector_material + ".exr")  # source / type / basename_material.exr
                motion_vector = load_motion_vector(path)
                save_as_fp16(os.path.join(tgt_dir, name), motion_vector)
                
            # Read and save no anti-aliasing frame.
            tgt_dir = check_dir(os.path.join(tgt_root, "LR_4x_NoAA"))
            for name in tqdm(frame_names, desc="LR_4x_NoAA"):
                # break
                path = os.path.join(src_root, dirname, name + frame_metarials + ".exr")
                frame = load_exr_with_rgb_chw(path)
                save_as_fp16(os.path.join(tgt_dir, name), frame)

        elif dirname == "LR_2x" or dirname == "LRx2":
            continue
            tasks: List[Process] = []

            # Get frame names without materials.
            materials = gbuffer_materials + [frame_metarials, motion_vector_material]
            frame_names = get_valid_frame_names(os.path.join(src_root, dirname), materials, shot_frames)

            # Read and save Gbuffers.
            tgt_dir = check_dir(os.path.join(tgt_root, "LR_2x_GBuffers"))
            def _sub_process_gbuffers(_frame_names):
                process_id = multiprocessing.current_process()._identity[0]
                for name in tqdm(_frame_names, desc=f"LR_2x_GBuffers: {str(process_id).zfill(2)}", ncols=150, nrows=24, position=process_id):
                    gbuffers = merge_gbuffers(os.path.join(src_root, dirname), name, gbuffer_materials, gbuffer_material_channles)
                    save_as_fp16(os.path.join(tgt_dir, name), gbuffers)
            
            for i in range(CPU_COUNT):
                _frame_names = frame_names[i::CPU_COUNT]
                task = Process(target=_sub_process_gbuffers, args=(_frame_names, ))
                task.start()
                tasks.append(task)
            for task in tasks: task.join()
            tasks.clear()

            # Reads and save motion vector.
            tgt_dir = check_dir(os.path.join(tgt_root, "LR_2x_MotionVector"))
            def _sub_process_velocity(_frame_names):
                process_id = multiprocessing.current_process()._identity[0]
                for name in tqdm(_frame_names, desc=f"LR_2x_MotionVector: {str(process_id).zfill(2)}", ncols=150, nrows=24, position=process_id):
                    path = os.path.join(src_root, dirname, name + motion_vector_material + ".exr")  # source / type / basename_material.exr
                    motion_vector = load_motion_vector(path)
                    save_as_fp16(os.path.join(tgt_dir, name), motion_vector)

            for i in range(CPU_COUNT):
                _frame_names = frame_names[i::CPU_COUNT]
                task = Process(target=_sub_process_velocity, args=(_frame_names, ))
                task.start()
                tasks.append(task)
            for task in tasks: task.join()
            tasks.clear()

            # Read and save no anti-aliasing frame.
            tgt_dir = check_dir(os.path.join(tgt_root, "LR_2x_NoAA"))
            def _sub_process_noaa(_frame_names):
                process_id = multiprocessing.current_process()._identity[0]
                for name in tqdm(_frame_names, desc=f"LR_2x_NoAA: {str(process_id).zfill(2)}", ncols=150, nrows=24, position=process_id):
                    path = os.path.join(src_root, dirname, name + frame_metarials + ".exr")
                    frame = load_exr_with_rgb_chw(path)
                    save_as_fp16(os.path.join(tgt_dir, name), frame)

            for i in range(CPU_COUNT):
                _frame_names = frame_names[i::CPU_COUNT]
                task = Process(target=_sub_process_noaa, args=(_frame_names, ))
                task.start()
                tasks.append(task)
            for task in tasks: task.join()
            tasks.clear()

        elif dirname == "NoAA_Others":
            tasks: List[Process] = []

            # Get frame names without materials.
            materials = [frame_metarials, motion_vector_material]
            frame_names = get_valid_frame_names(os.path.join(src_root, dirname), materials, shot_frames)

            # Read and save no anti-aliasing frame.
            tgt_dir = check_dir(os.path.join(tgt_root, "NoAA_HR"))
            def _sub_process_noaa(_frame_names):
                process_id = multiprocessing.current_process()._identity[0]
                for name in tqdm(_frame_names, desc=f"NoAA_HR: {str(process_id).zfill(2)}", ncols=150, nrows=24, position=process_id):
                    path = os.path.join(src_root, dirname, name + frame_metarials + ".exr")
                    frame = load_exr_with_rgb_chw(path)
                    save_as_fp16(os.path.join(tgt_dir, name), frame)

            for i in range(CPU_COUNT):
                _frame_names = frame_names[i::CPU_COUNT]
                task = Process(target=_sub_process_noaa, args=(_frame_names, ))
                task.start()
                tasks.append(task)
            for task in tasks: task.join()
            tasks.clear()

            # Reads and save motion vector.
            tgt_dir = check_dir(os.path.join(tgt_root, "NoAA_MotionVector"))
            def _sub_process_velocity(_frame_names):
                process_id = multiprocessing.current_process()._identity[0]
                for name in tqdm(_frame_names, desc=f"NoAA_MotionVector: {str(process_id).zfill(2)}", ncols=150, nrows=24, position=process_id):
                    path = os.path.join(src_root, dirname, name + motion_vector_material + ".exr")  # source / type / basename_material.exr
                    motion_vector = load_motion_vector(path)
                    save_as_fp16(os.path.join(tgt_dir, name), motion_vector)

            for i in range(CPU_COUNT):
                _frame_names = frame_names[i::CPU_COUNT]
                task = Process(target=_sub_process_velocity, args=(_frame_names, ))
                task.start()
                tasks.append(task)
            for task in tasks: task.join()
            tasks.clear()

        else:
            pass
        

if __name__ == "__main__":
    config = json.load(open("ue_data.json"))
    target_scenes = ["Bunker"]

    SPECULAR_VALID = True

    for scene_name, shot_frames in config["scenes"].items():
        if scene_name not in target_scenes:
            continue

        print("Scene: %s" % scene_name)
        src_root = os.path.join(config["original_data_path"], scene_name)
        tgt_root = os.path.join(config["compression_data_path"], scene_name)
        
        preprocess(src_root, tgt_root, shot_frames, config)

