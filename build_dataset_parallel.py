#!/usr/bin/env python
"""
Parallel dataset generation for AMN
Run from terminal: python build_dataset_parallel.py
"""

from multiprocessing import Pool
import sys
import os
from datetime import datetime

# Configuration
PARALLEL_LEVEL = 16

def create_random_medium_from_cobra(expname: str):    
    """
    Process a single experiment and generate training dataset
    Import heavy libraries only once per worker
    """
    
    # Import inside function to avoid pickling issues
    import pandas as pd
    from Library.Build_Dataset_lite import TrainingSet
    
    # Parameters
    cobraname = 'iML1515_duplicated_Lab_Data'
    mediumname = 'df_amn_dataset_levels'
    mediumbound = 'UB'
    exp_df_name = 'df_amn_dataset'
    method = 'pFBA'
    size_i = 100
    reduce = True
    verbose = True
    DIRECTORY = './'
    
    # Setup logging
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f'{log_dir}/{expname}_{timestamp}.log'
    
    log_f = open(log_file, 'w', buffering=1)
    
    def log(message):
        log_f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        log_f.flush()
    
    try:
        log(f"Starting processing for {expname}")
        
        # Get X from experimental data set
        cobrafile = DIRECTORY + 'Dataset_input/' + cobraname
        exp_data_path = f"H:/ROBOT_SCIENTIST/E_coli/Growth_rates/2025-10-31-27/processed/no_replicates/{expname}/AMN_dataset/"
        expfile = exp_data_path + exp_df_name

        log(f"Reading experimental data from {expfile}")
        df_exp = pd.read_csv(expfile + ".csv")
        mediumsize = len(df_exp.columns) - 1
        
        log(f"Creating TrainingSet with mediumsize={mediumsize}")
        parameter = TrainingSet(cobraname=cobrafile, 
                                mediumname=expfile, 
                                mediumbound=mediumbound, 
                                mediumsize=mediumsize, 
                                method='EXP', verbose=False)
        X = parameter.X.copy()
        log(f"X shape: {X.shape}")

        # Get other parameters from medium file
        mediumfile = exp_data_path + mediumname
        log(f"Reading medium file from {mediumfile}")
        parameter = TrainingSet(cobraname=cobrafile, 
                                mediumname=mediumfile, 
                                mediumbound=mediumbound, 
                                method=method, verbose=False)

        # Create varmed list
        log("Creating variable medium list")
        varmed = {}
        for i in range(X.shape[0]):
            varmed[i] = []
            for j in range(X.shape[1]):
                if parameter.levmed[j] > 1 and X[i, j] > 0:
                    varmed[i].append(parameter.medium[j])
        varmed = list(varmed.values())
        log(f"Variable medium created with {len(varmed)} entries")
        
        # Get COBRA training set
        log(f"Starting COBRA training set generation for {X.shape[0]} samples with size_i={size_i}")
        for i in range(X.shape[0]): 
            log(f"Processing sample {i+1}/{X.shape[0]}")
            # Pass log function to parameter.get so verbose output goes to log file
            parameter.get(sample_size=size_i, varmed=varmed[i], verbose=verbose, log_func=log) 
            log(f"Sample {i+1}/{X.shape[0]} completed")

        # Saving file
        trainingfile = DIRECTORY + 'Dataset_model/' + expname + '_' + parameter.mediumbound
        log(f"Saving training file to {trainingfile}")
        parameter.save(trainingfile, reduce=reduce)
        log(f"Successfully completed processing for {expname}")
        
        log_f.close()
        return f"{expname}: SUCCESS"
        
    except Exception as e:
        import traceback
        log(f"ERROR processing {expname}: {str(e)}")
        log(traceback.format_exc())
        log_f.close()
        return f"{expname}: FAILED - {str(e)}"


def main():
    """Main execution function"""
    # Create logs directory
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # Get list of experiments
    exp_dir = 'H:/ROBOT_SCIENTIST/E_coli/Growth_rates/2025-10-31-27/processed/no_replicates'
    expnames = os.listdir(exp_dir)

    print(f"Starting parallel processing of {len(expnames)} experiments with {PARALLEL_LEVEL} workers")
    print(f"Log files will be created in {os.path.abspath(log_dir)}/ directory")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Workers are being spawned...")
    print("(This may take 30-60 seconds for workers to load libraries)")
    sys.stdout.flush()

    # Run parallel processing
    with Pool(PARALLEL_LEVEL) as p:
        results = p.map(create_random_medium_from_cobra, expnames)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n=== Processing Summary ===")
    for result in results:
        print(result)
    print(f"\nCheck individual log files in {os.path.abspath(log_dir)}/ for detailed progress")


if __name__ == '__main__':
    main()
