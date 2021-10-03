## MAAD: A Model for Attended Awareness in Driving
[Install](#install) // [Datasets](#datasets) // [Training](#training) // [Experiments](#experiments) // [Analysis](#analysis) // [License](#license)

Official [PyTorch](https://pytorch.org/) implementation of *MAAD: A Model and Dataset for "Attended Awareness" in Driving* invented by the RAD Team at [Toyota Research Institute (TRI)](https://www.tri.global/), in particular for [Link to Paper](https://www.overleaf.com/project/5cb74be8dc8abc4e20cab4bc)
*Deepak Gopinath, Guy Rosman, Simon Stent, Katsuya Terahata, Luke Fletcher, Brenna Argall, John Leonard*.

MAAD affords estimation of attended awareness based on noisy gaze estimates and scene video over time. This learned model additionally affords saliency estimation and refinement of a noisy gaze signal. We demonstrate the performance of the model on a new, annotated dataset that explores the gaze and perceived attended awareness of subjects as they observe a variety of driving scenarios. In this dataset, we provide a surrogate annotated third person estimate of attended awareness as a reproducible supervisory cue.

## Install

You need a machine with recent Nvidia drivers and a GPU with at least 16GB of memory (more for the bigger models at higher resolution). We recommend using conda to have a reproducible environment. To setup your environment, type in a terminal (only tested in Ubuntu 18.04 and PyTorch 1.7.0):
```bash
git clone https://github.com/ToyotaResearchInstitute/att-aware.git
cd att-aware
# if you want to use conda (recommended)
conda env create -f environment.pt170.yml
conda activate pt170
```
We will list below all commands as if run directly inside the conda environment. If you encounter out of memory issues, try a lower `batch_size` parameter in the args_file.py.

## Datasets
All the datasets are assumed to be downloaded in `~/data/`.

### Videos
MAAD uses subset of videos (8 videos of urban driving) from th Dr(Eye)ve Dataset. The entire Dr(Eye)ve dataset can be downloaded at [Dr(Eye)ve Full Dataset](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=8). We collected gaze and attended awareness annotation data on the videos **[06, 07, 10, 11, 26, 35, 53, 60]**.
Each video folder should be located at `~/data/dreyeve/VIDEO_ID`

### Gaze Dataset
Our complete dataset comprises approximately 24.5 hours of gaze tracking data captured via multiple exposures from different subjects. We recruited 23 subjects (aged 20-55), who each watched a subset of video clips with their heads mounted in a chin-rest after a 9-point calibration procedure. Their primary task was to monitor the driving scene as a safety driver might monitor an autonomous vehicle. While not a perfect substitute for in-car driving data collection, this primary task allowed for the capture of many of the characteristics of attentive driving behavior. In order to explore the effect of the cognitive task difference (vs. in-car data) on the gaze and awareness estimates, subjects viewed the video under different cognitive task modifiers, as detailed in Section~\ref{sec:data:conditions} (data collected with non-null cognitive task modifiers comprise $30\%$ of total captured gaze data). Around $45\%$ of video stimuli were watched more than once, of which $11\%$ (40 minutes) was observed by 16 or more subjects.

The gaze dataset will be made available as a pkl (all_videos_subjects_tasks_gaze_data.pkl) file. Each subjects' gaze data is stored as a pandas dataframe in the pkl file (organized according to video, subject and task id). 
The pkl file is expected to be located at `~/data/all_videos_subjects_tasks_gaze_data.pkl`

### Attended Awareness Annotation Dataset
Our complete attended awareness annotation dataset consists of 54019 third-party annotations of approximately 10s long videos from the Gaze Dataset. Annotators watched a video snippet where the subject's gaze was marked by two circles centered at the gaze point. One circle (green) size was set to the diameter of a person's central foveal vision area at the viewing distance. Another circle (red) was set to a diameter twice the foveal vision circle. At the end of the video snippet, a specific location was chosen and the annotators were asked whether they believe the subject has attended to that location on a scale between 1 and 5 (1-no, definitely not aware, 5-yes, definitely aware). 
Each annotation consists of the following fields:
```bash
video_id | query_frame | subject | cognitive_modifier | query_x | query_y | anno_is_aware | anno_is_object | anno_expected_awareness | anno_surprise_factor
```
Any field which starts with `anno` is the annotation. For more details refer to supplementary material of the paper. 
Datasets are assumed to be downloaded in `~/data/datasets/MAAD_ATT_AWARENESS_LABELS.csv` (can be a symbolic link).

### Optic Flow
MAAD uses optic flow of the videos as a side-channel information to perform temporal regularizations. For the purposes of our model, we utilized [[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)] to generate optic flow. 
For each video in the dataset, the optic flow model has to be run all frame pairs N frames apart. The current code assumes that the optic flow generated is at half-resolution with a padding of 2 pixels (on each side) along the y direction. These parameters denoted as `OPTIC_FLOW_SCALE_FACTOR, OPTIC_FLOW_H_PAD, OPTIC_FLOW_W_PAD` can be altered in the `att-aware/src/maad/utils/maad_consts.py` file to suit your needs.

Optic flow is assumed to be cached as `~/maad_cache/optic_flow/VIDEO_ID/frame_N.npy`

### Segmentation Masks
MAAD uses segmentation masks for the videos in order to perform diffusivity-based spatial regularization. For the purposes of our model, we used MaskRCNN to generate the segmentation masks for each frame for each video. 

Segmentation masks are assumed to be cached as `~/maad_cache/segmentations_from_video/VIDEO_ID/segmentations_frames/frame_N.png`

During training, lower resolution mask images will be generated by resizing the full sized masks and will be cached back into the same location as `frame_N_ar_{aspect_ratio_reduction_factor}.png`.

## Training
MAAD model training can be done using the `train.py` script. 
Run the following command to train a model using all 8 videos (split into a train and test sets) using the parameter settings used in the ICCV paper. 
` python train.py --train_sequence_ids 6 7 10 11 26 35 53 60 --use_std_train_test_split --add_optic_flow --use_s3d --enable_amp`
Default resolution used is `240 x 135`.
All training args are present in `/att-aware/src/maad/config/args_file.py`

Models will be saved at `~/maad/models/TRAINING_HASH_NAME`
## Experiments
Three different experiments are proposed for MAAD. All experiments are done using the test split. Gaze Denoising and Awareness Estimation uses the trained model for inference. Gaze Calibration experiment involves continued training to optimize the miscalibration transform. 
All experiment results are saved as jsons in `~/maad/results/`
### Gaze Denoising
MAAD can be used for denoising noisy gaze estimates by relying on saliency information. 
The denoising experiment script is located at `att-aware/src/scripts/experiment_maad_denoising.py`

The script can be run using the following command:
`python experiment_maad_denoising.py --train_sequence_ids 6 7 10 11 26 35 53 60 --use_std_train_test_split --add_optic_flow --use_s3d --enable_amp --load_indices_dict_path ~/maad/logs/TRAINING_HASH/TRAINING_HASH/indices_dict_folder/indices_dict.pkl --load_model_path ~/maad/models/TRAINING_HASH/MODEL.pt --max_inference_num_batches 1000`

### Gaze Recalibration
MAAD can be used for recalibration of a miscalibrated gaze (due to errors in DMS). 
The calibration experiment script is located at `att-aware/src/scripts/experiment_maad_calibration.py`
The calibration experiment script can be run using the follow command:

`python experiment_maad_calibration_optimization.py --train_sequence_ids 6 7 10 11 26 35 53 60 --use_std_train_test_split --add_optic_flow --use_s3d --enable_amp --load_indices_dict_path ~/maad/logs/TRAINING_HASH/TRAINING_HASH/indices_dict_folder/indices_dict.pkl --load_model_path ~/maad/models/TRAINING_HASH/MODEL.pt --dropout_ratio '{"driver_facing":0.0, "optic_flow":0.0}'`

Note that, the above command assumes that the model used for recalibration was trained using the default cost parameters. It is important that the cost coefficients match the original values. Furthermore, the `dropout_ratio` for `driver_facing` gaze module should be set at `0.0` so that gaze is available as a side-channel input to the network at all times. The miscalibration noise levels can be specified using the `miscalibration_noise_levels` argument. 
 
### Awareness Estimation
MAAD can used for attended awareness estimation based on scene context and an imperfect gaze information. 
The attended awareness estimation script is located at `att-aware/src/scripts/experiment_maad_awareness_estimation.py`

The attended awareness estimation script can be run using the following command:
`python experiment_maad_awareness_estimation.py --train_sequence_ids 6 7 10 11 26 35 53 60 --use_std_train_test_split --add_optic_flow --use_s3d --enable_amp --load_indices_dict_path ~/maad/logs/TRAINING_HASH/TRAINING_HASH/indices_dict_folder/indices_dict.pkl --load_model_path ~/maad/models/TRAINING_HASH/MODEL.pt`

## Analysis
We have also provided scripts to parse and compute statistics on the results outputted by the experiment scripts. These scripts are available at `att-aware/src/scripts/parse_*_data.py` where `*` could be `denoising, calibration_optimization, awareness_estimation`

The results of the parsing scripts will be outputted directly in the terminal. The parsing scripts can be run using the following commands.
`python parse_denoising_data.py --results_json_prefix ~/maad/results/GAZE_DENOISING`. Assumes that the result of the denoising experiment is in `GAZE_DENOISING.json`

`python parse_awareness_estimation_data.py --results_json_prefix ~/maad/results/AWARENESS_ESTIMATION`. Assumes that the result of the awareness estimation experiment is in `AWARENESS_ESTIMATION.json`

The results of the calibration experiments are expected to stored in files with the following filename convention
`experiment_type_gaze_calibration_miscalibration_noise_level_NOISELEVEL_optimization_run_num_OPTIMIZATIONNUM_FILENAMEAPPEND.json`,
where `NOISELEVEL` is in the `miscalibration_noise_levels` argument in `experiment_maad_calibration_optimization.py`
`OPTIMIZATIONNUM` goes from 0 to `num_optimization_runs`-1 and `FILENAMEAPPEND` is the `filename_append` argument in the experiment.

`python parse_calibration_optimization_data.py --folder_containing_results FOLDER_CONTAINING_JSONS --num_optimization_runs (same val as used in the experiment) --miscalibration_noise_levels (same val as used in the experiment) --filename_append (same val as used in the experiment)`

## License
The source code is released under the [MIT License](LICENSE.md)
