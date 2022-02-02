import hashlib
import os
import uuid

import cv2
import numpy as np

from maad.configs.args_file import parse_arguments

DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

"""
Assumes that generate_saliency_maps_using_model.py has already been run to generate the saliency maps. 
Going with the same examples of video_id=6, task_id=control, subject_id=2 from generate_saliency_maps_using_model.py the 
example usage of this script will be 

python generate_video_from_saliency_maps.py --train_sequence_ids 6
--saliency_maps_directory PATH_TO_DIRECTORY_CONTAINING_THE_SALIENCY_MAP_OUTPUTS
(--generate_original_scale_video)
"""


def generate_video_from_saliency_maps(args):
    assert args.data_dir != "None"
    assert args.precache_dir != "None"
    assert args.saliency_maps_directory != None, "Needs the saliency map directory to generate videos"

    assert os.path.exists(args.saliency_maps_directory)
    assert len(args.train_sequence_ids) == 1  # can only generate one video at a time
    assert args.start_index_percentage >= 0.0 and args.start_index_percentage < 100.0
    assert args.end_index_percentage > 0.0 and args.end_index_percentage <= 100.0
    assert args.end_index_percentage > args.start_index_percentage

    start_index_percentage = args.start_index_percentage
    end_index_percentage = args.end_index_percentage

    video_desc_string = args.video_desc_string
    saliency_maps_directory = args.saliency_maps_directory  # this is a full path
    video_id = args.train_sequence_ids[0]
    video_filename = os.path.join(args.data_dir, "{0:02d}".format(video_id), "video_garmin.avi")
    frame_image_folder_path = os.path.join(args.precache_dir, "frame_image_cached")
    aspect_ratio_reduction_factor = args.aspect_ratio_reduction_factor

    reduced_video_width = int(round(DISPLAY_WIDTH / aspect_ratio_reduction_factor))
    reduced_video_height = int(round(DISPLAY_HEIGHT / aspect_ratio_reduction_factor))

    hmap_videos_directory = os.path.join(args.log_dir, "hmap_videos_directory")
    os.makedirs(hmap_videos_directory, exist_ok=True)

    # if this is true the saliency maps generated needs to upsampled by the ar_reduction factor
    # so that it matches the original dimension. If false, then the reduced size road image will be used for generating video.
    generate_original_scale_video = args.generate_original_scale_video

    all_saliency_maps = os.listdir(saliency_maps_directory)
    assert len(all_saliency_maps) > 0  # make sure the saliency maps have been created

    # sorted all the saliency maps (with and without gaze) and awareness maps (with and without gaze)
    gaze_hmap_with_gaze_list = sorted([s for s in all_saliency_maps if s.find("gaze_hmap_with_gaze") != -1])
    gaze_hmap_without_gaze_list = sorted([s for s in all_saliency_maps if s.find("gaze_hmap_without_gaze") != -1])
    awareness_hmap_with_gaze_list = sorted([s for s in all_saliency_maps if s.find("awareness_hmap_with_gaze") != -1])
    awareness_hmap_without_gaze_list = sorted(
        [s for s in all_saliency_maps if s.find("awareness_hmap_without_gaze") != -1]
    )

    # sanity checks to make sure the list is not empty and have equal length
    assert len(gaze_hmap_with_gaze_list) > 0
    assert (
        len(gaze_hmap_with_gaze_list)
        == len(gaze_hmap_without_gaze_list)
        == len(awareness_hmap_with_gaze_list)
        == len(awareness_hmap_without_gaze_list)
    )

    # note that the following variables are extracted assuming that the maps were created using generate_saliency_maps_using_model.py script.
    # the images are saved according to a predefined naming convention
    video_id_string = gaze_hmap_with_gaze_list[0][:11]  # for example 'VIDEO_ID_06'
    frame_idxs = [int(f[22:30]) for f in gaze_hmap_with_gaze_list]

    start_frame_idx = frame_idxs[int((start_index_percentage / 100.0) * (len(frame_idxs) - 1))]
    end_frame_idx = frame_idxs[int((end_index_percentage / 100.0) * (len(frame_idxs) - 1))]

    # list of full paths to all of the saliency maps
    gaze_hmap_with_gaze_list_full_path_list = [
        os.path.join(saliency_maps_directory, s) for s in gaze_hmap_with_gaze_list
    ]
    gaze_hmap_without_gaze_list_full_path_list = [
        os.path.join(saliency_maps_directory, s) for s in gaze_hmap_without_gaze_list
    ]
    awareness_hmap_with_gaze_list_full_path_list = [
        os.path.join(saliency_maps_directory, s) for s in awareness_hmap_with_gaze_list
    ]
    awareness_hmap_without_gaze_list_full_path_list = [
        os.path.join(saliency_maps_directory, s) for s in awareness_hmap_without_gaze_list
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25  # 7500 frames in 5 minutes for original video

    if generate_original_scale_video:
        print("ORIGINAL SCALE")
        gaze_hmap_with_gaze_video_filename = os.path.join(
            hmap_videos_directory,
            video_id_string + "_" + video_desc_string + "_gaze_hmap_with_gaze_video_original_res.mp4",
        )
        gaze_hmap_with_gaze_write_cap = cv2.VideoWriter(
            gaze_hmap_with_gaze_video_filename, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )
        gaze_hmap_without_gaze_video_filename = os.path.join(
            hmap_videos_directory,
            video_id_string + "_" + video_desc_string + "_gaze_hmap_without_gaze_video_original_res.mp4",
        )
        gaze_hmap_without_gaze_write_cap = cv2.VideoWriter(
            gaze_hmap_without_gaze_video_filename, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        awareness_hmap_with_gaze_video_filename = os.path.join(
            hmap_videos_directory,
            video_id_string + "_" + video_desc_string + "_awareness_hmap_with_gaze_video_original_res.mp4",
        )
        awareness_hmap_with_gaze_write_cap = cv2.VideoWriter(
            awareness_hmap_with_gaze_video_filename, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )

        awareness_hmap_without_gaze_video_filename = os.path.join(
            hmap_videos_directory,
            video_id_string + "_" + video_desc_string + "_awareness_hmap_without_gaze_video_original_res.mp4",
        )
        awareness_hmap_without_gaze_write_cap = cv2.VideoWriter(
            awareness_hmap_without_gaze_video_filename, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
        )
    else:
        print("REDUCED SCALE")
        gaze_hmap_with_gaze_video_filename = os.path.join(
            hmap_videos_directory,
            video_id_string
            + "_"
            + video_desc_string
            + "_gaze_hmap_with_gaze_video_ar_{}_res.mp4".format(aspect_ratio_reduction_factor),
        )
        gaze_hmap_with_gaze_write_cap = cv2.VideoWriter(
            gaze_hmap_with_gaze_video_filename, fourcc, fps, (reduced_video_width, reduced_video_height)
        )
        gaze_hmap_without_gaze_video_filename = os.path.join(
            hmap_videos_directory,
            video_id_string
            + "_"
            + video_desc_string
            + "_gaze_hmap_without_gaze_video_ar_{}_res.mp4".format(aspect_ratio_reduction_factor),
        )
        gaze_hmap_without_gaze_write_cap = cv2.VideoWriter(
            gaze_hmap_without_gaze_video_filename, fourcc, fps, (reduced_video_width, reduced_video_height)
        )

        awareness_hmap_with_gaze_video_filename = os.path.join(
            hmap_videos_directory,
            video_id_string
            + "_"
            + video_desc_string
            + "_awareness_hmap_with_gaze_video_ar_{}_res.mp4".format(aspect_ratio_reduction_factor),
        )
        awareness_hmap_with_gaze_write_cap = cv2.VideoWriter(
            awareness_hmap_with_gaze_video_filename, fourcc, fps, (reduced_video_width, reduced_video_height)
        )

        awareness_hmap_without_gaze_video_filename = os.path.join(
            hmap_videos_directory,
            video_id_string
            + "_"
            + video_desc_string
            + "_awareness_hmap_without_gaze_video_ar_{}_res.mp4".format(aspect_ratio_reduction_factor),
        )
        awareness_hmap_without_gaze_write_cap = cv2.VideoWriter(
            awareness_hmap_without_gaze_video_filename, fourcc, fps, (reduced_video_width, reduced_video_height)
        )

    blend_alpha = 0.6
    # sorted list of available frame indices.
    for (
        frame_idx,
        gaze_hmap_with_gaze_for_frame_idx_filename,
        gaze_hmap_without_gaze_for_frame_idx_filename,
        awareness_hmap_with_gaze_for_frame_idx_filename,
        awareness_hmap_without_gaze_for_frame_idx_filename,
    ) in zip(
        frame_idxs,
        gaze_hmap_with_gaze_list_full_path_list,
        gaze_hmap_without_gaze_list_full_path_list,
        awareness_hmap_with_gaze_list_full_path_list,
        awareness_hmap_without_gaze_list_full_path_list,
    ):
        if frame_idx < start_frame_idx:
            continue
        if frame_idx > end_frame_idx:
            break

        print("Writing frame id ", frame_idx)
        # read in the hmaps
        gaze_hmap_with_gaze_frame = cv2.imread(gaze_hmap_with_gaze_for_frame_idx_filename)  # (h, w, 3), bgr
        gaze_hmap_without_gaze_frame = cv2.imread(gaze_hmap_without_gaze_for_frame_idx_filename)
        awareness_hmap_with_gaze_frame = cv2.imread(awareness_hmap_with_gaze_for_frame_idx_filename)
        awareness_hmap_without_gaze_frame = cv2.imread(awareness_hmap_without_gaze_for_frame_idx_filename)

        full_size_precache_id = "frame_{}".format(frame_idx)
        full_size_precache_filename = (
            os.path.join(frame_image_folder_path, "{0:02d}".format(video_id), full_size_precache_id) + ".jpg"
        )
        assert os.path.exists(
            full_size_precache_filename
        ), "Full resolution video frames need to be extracted and cached beforehand"
        reduced_size_precache_id = full_size_precache_id + "_ar_{}".format(aspect_ratio_reduction_factor)

        # full path to reduced_size image (in the same folder as the full size image)
        reduced_size_precache_filename = (
            os.path.join(frame_image_folder_path, "{0:02d}".format(video_id), reduced_size_precache_id) + ".jpg"
        )
        if os.path.exists(reduced_size_precache_filename):
            road_frame = cv2.imread(reduced_size_precache_filename)
        else:
            full_frame = cv2.imread(full_size_precache_filename)
            # resize full frame and cache the result in the same folder for later use.
            full_frame = np.float32(full_frame)
            road_frame = cv2.resize(full_frame, (reduced_video_width, reduced_video_height))
            # cache reduced size frame for future use.
            cv2.imwrite(reduced_size_precache_filename, road_frame)

        if generate_original_scale_video:
            assert os.path.exists(full_size_precache_filename)
            # in bgr in uint8 (H, W, 3)
            road_frame = cv2.imread(full_size_precache_filename)
            # resize the hmaps to original size before blending. All bgr
            gaze_hmap_with_gaze_frame = cv2.resize(gaze_hmap_with_gaze_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))  # BGR,
            gaze_hmap_without_gaze_frame = cv2.resize(gaze_hmap_without_gaze_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            awareness_hmap_with_gaze_frame = cv2.resize(awareness_hmap_with_gaze_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            awareness_hmap_without_gaze_frame = cv2.resize(
                awareness_hmap_without_gaze_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)
            )
            assert (
                gaze_hmap_with_gaze_frame.shape[0]
                == gaze_hmap_without_gaze_frame.shape[0]
                == awareness_hmap_with_gaze_frame.shape[0]
                == awareness_hmap_without_gaze_frame.shape[0]
                == road_frame.shape[0]
                == DISPLAY_HEIGHT
            )
            assert (
                gaze_hmap_with_gaze_frame.shape[1]
                == gaze_hmap_without_gaze_frame.shape[1]
                == awareness_hmap_with_gaze_frame.shape[1]
                == awareness_hmap_without_gaze_frame.shape[1]
                == road_frame.shape[1]
                == DISPLAY_WIDTH
            )
        else:
            assert (
                gaze_hmap_with_gaze_frame.shape[0]
                == gaze_hmap_without_gaze_frame.shape[0]
                == awareness_hmap_with_gaze_frame.shape[0]
                == awareness_hmap_without_gaze_frame.shape[0]
                == road_frame.shape[0]
                == reduced_video_height
            )
            assert (
                gaze_hmap_with_gaze_frame.shape[1]
                == gaze_hmap_without_gaze_frame.shape[1]
                == awareness_hmap_with_gaze_frame.shape[1]
                == awareness_hmap_without_gaze_frame.shape[1]
                == road_frame.shape[1]
                == reduced_video_width
            )

        # blend heatmaps and video frames
        overlaid_gaze_hmap_with_gaze_frame = blend_alpha * np.float32(road_frame / 255) + (
            1 - blend_alpha
        ) * np.float32(gaze_hmap_with_gaze_frame / 255)
        overlaid_gaze_hmap_without_gaze_frame = blend_alpha * np.float32(road_frame / 255) + (
            1 - blend_alpha
        ) * np.float32(gaze_hmap_without_gaze_frame / 255)
        overlaid_awareness_hmap_with_gaze_frame = blend_alpha * np.float32(road_frame / 255) + (
            1 - blend_alpha
        ) * np.float32(awareness_hmap_with_gaze_frame / 255)
        overlaid_awareness_hmap_without_gaze_frame = blend_alpha * np.float32(road_frame / 255) + (
            1 - blend_alpha
        ) * np.float32(awareness_hmap_without_gaze_frame / 255)

        # back to uint8 for cv2 writing
        overlaid_gaze_hmap_with_gaze_frame = np.array(overlaid_gaze_hmap_with_gaze_frame * 255, dtype=np.uint8)
        overlaid_gaze_hmap_without_gaze_frame = np.array(overlaid_gaze_hmap_without_gaze_frame * 255, dtype=np.uint8)
        overlaid_awareness_hmap_with_gaze_frame = np.array(
            overlaid_awareness_hmap_with_gaze_frame * 255, dtype=np.uint8
        )
        overlaid_awareness_hmap_without_gaze_frame = np.array(
            overlaid_awareness_hmap_without_gaze_frame * 255, dtype=np.uint8
        )

        # write blended image into video handle.
        gaze_hmap_with_gaze_write_cap.write(overlaid_gaze_hmap_with_gaze_frame)
        gaze_hmap_without_gaze_write_cap.write(overlaid_gaze_hmap_without_gaze_frame)
        awareness_hmap_with_gaze_write_cap.write(overlaid_awareness_hmap_with_gaze_frame)
        awareness_hmap_without_gaze_write_cap.write(overlaid_awareness_hmap_without_gaze_frame)

    gaze_hmap_with_gaze_write_cap.release()
    gaze_hmap_without_gaze_write_cap.release()
    awareness_hmap_with_gaze_write_cap.release()
    awareness_hmap_without_gaze_write_cap.release()

    print("DONE MAKING VIDEOS")
    print("VIDEOS in ", hmap_videos_directory)


def arg_setter(parser):
    parser.add_argument(
        "--saliency_maps_directory",
        dest="saliency_maps_directory",
        default=None,
        help="The directory where the saliency maps are stored precached data is",
    )

    parser.add_argument(
        "--generate_original_scale_video",
        action="store_true",
        default=False,
        help="Flag for using generating original scale video from saliency maps",
    )

    parser.add_argument(
        "--start_index_percentage",
        action="store",
        type=float,
        default=0.0,
        help="The starting point of the saliency video (in percentage)",
    )

    parser.add_argument(
        "--end_index_percentage",
        action="store",
        type=float,
        default=100.0,
        help="The end point of the saliency video (in percentage)",
    )
    parser.add_argument(
        "--video_desc_string",
        dest="video_desc_string",
        default="saliency_video",
        help="meaningful descriptor string for videos generated from saliency maps (typically encoding the model info)",
    )


if __name__ == "__main__":
    session_hash = uuid.uuid4().hex
    args = parse_arguments(session_hash, [arg_setter])
    generate_video_from_saliency_maps(args)
