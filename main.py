import argparse
import asyncio
import csv
import os
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
from preds import make_predictions, make_ref, move_preds, move_ref, read_predictions, write_predictions

from utils import copy_results, create_database, create_yaml, launch_finetuning, namespace_to_dict, save_json


def parse_opt(known=False):
    parser = argparse.ArgumentParser("Evaluate the performance of a Cycle GAN trained, steps:\n\
                                     1. Train a Cycle GAN with some parameters\n\
                                     2. Generate a new base with the generator of this CG trained\n\
                                     3. Create a new YAML in order to train a YOLOv5 on this new base\n\
                                     4. Train the YOLOv5\n\
                                     5. Use AL of these elements to evaluate performance")
    
    # Global
    parser.add_argument('--debug', action='store_true', help="To debug the code")
    parser.add_argument('--device', default='cpu', help='cuda device (i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--everything', action='store_true',
                        help='To make everything in the code (predictions, create a finetuning and launch it, create the new YAML')
    parser.add_argument('--finetuning', action='store_true', help='To finetune the old model')
    parser.add_argument('--k_folds', action='store_true', help='To learn on a changing validation folder')
    parser.add_argument('--vary_src_img', action='store_true', help='To vary the number of source images')
    parser.add_argument('--server', default='DGX', help="Choose between: 'DGX' or 'JZ' (default=DGX)")


    # Parameters for the run
    parser.add_argument('--nb_img_dgx', type=int, default=50, help='Number of images of DGX database to integrate during the active learning')
    parser.add_argument('--nb_img_src', type=int, default=1000,
                        help='Number of images of SOURCE database to integrate during the active learning')
    parser.add_argument('--project', type=str, help='To name the folder which will contains all predictions',
                        default='Yolo_preds')
    parser.add_argument('--finetuning_dir', type=str, help='Output directory for models finetuned',
                        default='/mnt/MO-DGX/storage_thesis_hr/Results_HR/Active_Learning/Finetuning')
    parser.add_argument('--preds_dir', type=str, help='Output directory for models finetuned',
                        default='/mnt/MO-DGX/storage_thesis_hr/Results_HR/Active_Learning/Predictions')
    parser.add_argument('--output_dir', type=str, help='Folder where all results will be save',
                        default="/mnt/MO-DGX/storage_thesis_hr/Results_HR/Active_Learning/AL")
    parser.add_argument('--reference_dir', type=str, help='Folder where reference results will be save',
                        default="/mnt/MO-DGX/storage_thesis_hr/Results_HR/Active_Learning/Reference")
    parser.add_argument('--img_dgx_dir', type=str, help='Folder which contains all DGX images to be tested',
                        default="/mnt/MO-DGX/storage_thesis_hr/OLD_DGX/CROSS_VALIDATION/CROSS_VALIDATION_512/images")
    parser.add_argument('--img_src_dir', type=str, help='Folder which contains all SOURCE images to be tested',
                        default="/raid/SIMSON_512/images")
        
    # Parameters for Yolo
    parser.add_argument('--only_preds', action='store_true', help='To make only predictions (default: all is done except preds)')
    parser.add_argument('--make_preds', action='store_true', help='To make the predictions by the Yolo model')
    parser.add_argument('--make_preds_dgx', action='store_true',
                        help='To make ONLY predictions by the Yolo model for DGX images')
    parser.add_argument('--make_preds_src', action='store_true',
                        help='To make ONLY predictions by the Yolo model for SIMSON images')
    parser.add_argument('--yolo_model', type=str, help='Path to the Yolo model (.pt file)',
                        default="/home/hruiz/codes/yolov5/SIMSON/Training_100_L_onDGX/weights/best.pt")
    parser.add_argument('--yolo_dir', type=str, help='Path to the Yolo directory', default="/home/hruiz/codes/yolov5")
    parser.add_argument('--conf_thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--img_size', type=int, default=512, help='Dimension of images')
    
    # New database with worst images
    parser.add_argument('--create_database', action='store_true', help='To create a new database with only symbolic links')
    parser.add_argument('--create_finetune', action='store_true', help='To create a finetune which contains worst images to train in active learning')
    parser.add_argument('--create_yaml', action='store_true', help='To create a YAML file with new images to give to Yolo')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
    

async def main(opt):
    model_name = opt.yolo_model.split('/')[-3]
    script_dir = Path(__file__).parent
    k_folds_dir = "Kfolds" if opt.k_folds else "no_Kfolds"
    opt.finetuning_dir = Path(opt.finetuning_dir).joinpath(k_folds_dir, f"{opt.nb_img_src}_src_img", model_name)
    opt.preds_dir = Path(opt.preds_dir).joinpath(model_name)
    opt.duration_dir = opt.output_dir.joinpath("Durations")
    opt.output_dir = Path(opt.output_dir).joinpath(k_folds_dir, f"{opt.nb_img_src}_src_img", model_name)
    opt.reference_dir = Path(opt.reference_dir).joinpath(k_folds_dir, model_name)
    results_op_dir = Path(opt.output_dir).joinpath("Results")
    results_op_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments into a json file
    opt.output_param = opt.output_dir.joinpath("Parameters", f"Parameters_{opt.nb_img_dgx:03}.json")
    Path(opt.output_param).parent.mkdir(parents=True, exist_ok=True)
    opt_dict = namespace_to_dict(opt)
    save_json(opt_dict, opt.output_param)
    
    # Predictions
    t_begin_preds = datetime.now()
    if opt.make_preds or opt.make_preds_dgx or opt.make_preds_src:
        print("\n" + "="*10 + " Begin of predictions " + "="*10 + "\n")
        opt.preds_dir.mkdir(parents=True, exist_ok=True)
        # DGX images
        if opt.make_preds_dgx or opt.make_preds:
            for i in range(4):
                make_predictions(opt, f"{opt.img_dgx_dir}/{i+1}", f"DGX_Preds_{i+1}")
        # SOURCE images
        src_name = Path(opt.img_src_dir).parts[-2]
        if opt.make_preds_src or opt.make_preds:
            make_predictions(opt, opt.img_src_dir, f"Preds_{src_name}")
        print("\n" + "="*10 + " End of predictions " + "="*10 + "\n")

        print("="*10 + f" Move predictions " + "="*10)
        print(f"Output directory: {opt.output_dir}")
        # Move these predictions into the output_dir with the correct project name
        preds_fold = script_dir.joinpath(opt.project)
        preds_paths = [f for f in preds_fold.glob("**/*.txt")] # Get all predictions paths
        # Move them into the fianl output directory
        await asyncio.gather(move_preds(p, opt.preds_dir, src_name) for p in preds_paths)
        copy_results(opt.project, model_name, results_op_dir, yolo_dir=opt.yolo_dir)
        shutil.rmtree(preds_fold)
        print("="*10 + f" End moving predictions " + "="*10)
    t_end_preds = datetime.now()
    duration_preds = t_end_preds - t_begin_preds

    # K folds
    t_begin_kfolds = datetime.now()
    nb_iter = 4 if opt.k_folds else 1 # Number of folders in DGX database that we use
    duration_df = pd.DataFrame(index=[f"CV{i+1}" for i in range(nb_iter)], columns=["Finetune_files", "Database", "YAML", "Reference", "Finetuning"])
    for i in range(nb_iter):
        name_cv = f"CV{i+1}" if nb_iter > 1 else "CVall"

        # Create a file to finetune the model
        t_begin_finetune_files = datetime.now()
        finetune_op_dir = opt.output_dir.joinpath("Nb_images_reinjected", str(opt.nb_img_dgx), name_cv)
        finetune_op_dir.mkdir(parents=True, exist_ok=True)
        finetune_file = finetune_op_dir.joinpath(f"TRN_{model_name}_{name_cv}_ActiveLearning.txt")
        if opt.create_finetune:
            print("\n" + "="*10 + f" Begin creation of finetuning file '{name_cv}' " + "="*10)
            print(f"File: {finetune_file}")
            print(f"Read predictions in: {opt.preds_dir}")
            dgx_preds_df, src_preds_df = await read_predictions(opt.preds_dir) # Get dataframes
            # DGX
            dgx_files = list(dgx_preds_df['file']) # Get names
            dgx_files = list(dict.fromkeys(dgx_files)) # Erase duplicate
            all_dgx = f"{opt.img_dgx_dir}/all"
            # SOURCE
            src_files = list(src_preds_df['file']) # Get names
            src_files = list(dict.fromkeys(src_files)) # Erase duplicate
            all_src = opt.img_src_dir
            # Paramaters
            nb_imgs = (opt.nb_img_dgx, opt.nb_img_src)
            imgs = (dgx_files, src_files)
            all_src = (all_dgx, all_src)
            write_predictions(nb_imgs, imgs, all_src, finetune_file, name_cv)
            print(f"File '{finetune_file}' created")
        t_end_finetune_files = datetime.now()
        delta_finetune_files = t_end_finetune_files - t_begin_finetune_files
        duration_df["Finetune_files"][i] = delta_finetune_files.total_seconds()

        # Create a new database with only interesting files
        t_begin_yaml = datetime.now()
        if opt.create_database:
            print("\n" + "="*10 + " Begin creation of database " + "="*10)
            database_op_dir = finetune_op_dir.joinpath("Database")
            await create_database(finetune_file, database_op_dir)
            print("\n" + "="*10 + " End creation of database " + "="*10)
        t_end_yaml = datetime.now()
        delta_databse = t_end_yaml - t_begin_yaml
        duration_df["Database"][i] = delta_databse.total_seconds()
        
        # Create YAML file to give for the new training
        t_begin_yaml = datetime.now()
        yaml_file = finetune_op_dir.joinpath(f"{model_name}_ActiveLearning.yaml")
        if opt.create_yaml:
            create_yaml(yaml_file, finetune_file)
            print(f"File '{yaml_file}' created")
        t_end_yaml = datetime.now()
        delta_yaml = t_end_yaml - t_begin_yaml
        duration_df["YAML"][i] = delta_yaml.total_seconds()

        # Create reference results if they don't already exists
        t_begin_ref = datetime.now()
        ref_dir = opt.reference_dir.joinpath(name_cv)
        if opt.make_ref and not ref_dir.exists():
            print("\n" + "="*10 + " Begin creation of reference " + "="*10)
            print(f"Directory: {ref_dir}")
            ref_dir.mkdir(parents=True, exist_ok=True)
            make_ref(opt, yaml_file, name_cv)
            shutil.move(f"Ref_{opt.project}/{name_cv}/PR_curve.png", ref_dir)
            shutil.rmtree(f"Ref_{opt.project}")
            print("\n" + "="*10 + " End creation of reference " + "="*10)
        t_end_ref = datetime.now()
        delta_ref = t_end_ref - t_begin_ref
        duration_df["Reference"][i] = delta_ref.total_seconds()

        # Launch finetuning of the model
        t_begin_finetuning = datetime.now()
        if opt.finetuning and not opt.debug:
            print("\n" + "="*10 + " Begin launch finetuning " + "="*10)
            launch_finetuning(opt, yaml_file, model_name, opt.nb_img_dgx, name_cv, verbose=opt.debug)
            print("\n" + "="*10 + " End launch finetuning " + "="*10)
            
            print("\n" + "="*10 + " Begin copying results " + "="*10)
            copy_results(opt.finetuning_dir, model_name, results_op_dir, nb_img=opt.nb_img_dgx, cv=name_cv, verbose=opt.debug)
            print("\n" + "="*10 + " End copying results " + "="*10)
        t_end_finetuning = datetime.now()
        delta_finetuning = t_end_finetuning - t_begin_finetuning
        duration_df["Finetuning"][i] = delta_finetuning.total_seconds()

    t_end_kfolds = datetime.now()
    duration_kfolds = t_end_kfolds - t_begin_kfolds

    print(f"Predictions time: {duration_preds}")
    print(f"Kfolds total time: {duration_kfolds}")
    print(duration_df)

    # Write duration of each parts
    if opt.only_preds:
        duration_file = opt.duration_dir.joinpath(f"Preds_{Path(opt.img_src_dir).parts[-2]}.csv")
    else:
        if opt.vary_src_img:
            duration_file = opt.duration_dir.joinpath(k_folds_dir,
                                                      f"{model_name}_varySRC",
                                                      f"duration_{opt.nb_img_src:04}.csv")
        else:
            duration_file = opt.duration_dir.joinpath(k_folds_dir,
                                                      f"{model_name}",
                                                      f"duration_{opt.nb_img_dgx:03}.csv")
    duration_file.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "="*10 + " Begin writing duration times " + "="*10)
    print(f"File: {duration_file}")
    duration_df.loc['Total'] = duration_df.sum()
    duration_df['Total'] = duration_df.sum(axis=1)
    duration_df.to_csv(duration_file)
    print("\n" + "="*10 + " End writing duration times " + "="*10)

    print('\n')
    print("="*41)
    print("="*15 + f" FINAL END " + "="*15)
    print("="*41)
    

if __name__ == '__main__':
    opt = parse_opt()

    # Change some directories to avoid conflicts during DEBUG
    if opt.debug:
        opt.project = f"DEBUG_{opt.project}"
        opt.output_dir = Path(opt.output_dir).parent.joinpath("DEBUG", "AL")
        opt.reference_dir = Path(opt.reference_dir).parent.joinpath("DEBUG", "Reference")
        opt.finetuning_dir = Path(opt.finetuning_dir).parent.joinpath("DEBUG", "Finetuning")

    if opt.everything:
        opt.make_preds = True
        opt.make_ref = True
        opt.create_finetune = True
        opt.create_yaml = True
        opt.finetuning = True
    elif opt.only_preds:
        opt.make_preds = True
        opt.make_ref = False
        opt.create_finetune = False
        opt.create_yaml = False
        opt.finetuning = False
    else:
        opt.make_preds = False
        opt.make_ref = True
        opt.create_finetune = True
        opt.create_yaml = True
        opt.finetuning = True

    if opt.server == "JZ":
        try:
            # slurm_job_id = int(os.environ["SLURM_JOB_ID"])
            slurm_array_job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])  # "305813"_2
            slurm_array_task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])  # 305813_"2"
            # slurm_array_task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
            # slurm_array_task_min = int(os.environ["SLURM_ARRAY_TASK_MIN"])
            # slurm_array_task_max = int(os.environ["SLURM_ARRAY_TASK_MAX"])
        
            # Number of images to reinjected in the active learning
            if opt.vary_src_img:
                opt.nb_img_src = slurm_array_task_id
            else:
                opt.nb_img_dgx = slurm_array_task_id
        except:
            pass
    
    asyncio.run(main(opt))
