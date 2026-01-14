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

def create_random_medium_from_cobra(strain_name: str):    
    """
    Process a single experiment and generate training dataset
    Import heavy libraries only once per worker
    """
    
    # Import inside function to avoid pickling issues
    import pandas as pd
    from Library.Build_Dataset_lite import TrainingSet
    
    # Parameters
    cobraname = 'iML1515_duplicated_Lab_Data'
    mediumname = 'df_AMN_level'
    mediumbound = 'UB'
    exp_df_name = 'df_AMN_input'
    method = 'pFBA'
    size_i = 1
    test_ratio = 0.2
    reduce = False
    verbose = True
    DIRECTORY = '../../'
    
    # Setup logging
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_file = f'{log_dir}/{strain_name}_{timestamp}.log'
    
    log_f = open(log_file, 'w', buffering=1)
    
    def log(message):
        log_f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        log_f.flush()
    
    try:
        log(f"Starting processing for {strain_name}")
        
        # Get X from experimental data set
        cobrafile = DIRECTORY + 'Dataset_input/' + cobraname
        exp_data_path = f"H:/ROBOT_SCIENTIST/E_coli/Growth_rates/2025-10-31-27/processed/replicates_STRAINS/{strain_name}/AMN_dataset/"
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
            log(f"varmed[{i}]: {varmed[i]}")
            parameter.get(sample_size=size_i, varmed=varmed[i], verbose=verbose, log_func=log) 
            log(f"Sample {i+1}/{X.shape[0]} completed")
        # Saving file
        trainingfile = DIRECTORY + 'Dataset_model/' + strain_name + '_' + parameter.mediumbound + '_' + str(size_i)
        log(f"Saving training file to {trainingfile}")
        log("Step 1/3: Updating stoichiometric matrices...")
        parameter.update_matrices(verbose=False, log_func=log)
        log("Step 2/3: Updating LP matrices (this may take several minutes for large datasets)...")
        parameter.update_matrices_LP(verbose=False, log_func=log)
        log("Step 3/3: Writing files to disk...")
        parameter.save(trainingfile, reduce=reduce, log_func=log)
        log(f"Successfully completed processing for {strain_name}")
        
        log_f.close()
        log(f"Y : {parameter.Y}   ")
        return f"{strain_name}: SUCCESS"
        
    except Exception as e:
        import traceback
        log(f"ERROR processing {strain_name}: {str(e)}")
        log(traceback.format_exc())
        log_f.close()
        return f"{strain_name}: FAILED - {str(e)}"


def main():
    """Main execution function"""
    # Create logs directory
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    # Get list of experiments
    exp_dir = 'H:/ROBOT_SCIENTIST/E_coli/Growth_rates/2025-10-31-27/processed/replicates_STRAINS'
    strains_name = os.listdir(exp_dir)
    strains_name = [name for name in strains_name if os.path.isdir(os.path.join(exp_dir, name))]

    print(f"Starting parallel processing of {len(strains_name)} experiments with {PARALLEL_LEVEL} workers")
    print(f"Log files will be created in {os.path.abspath(log_dir)}/ directory")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Workers are being spawned...")
    print("(This may take 30-60 seconds for workers to load libraries)")
    sys.stdout.flush()

    # Run parallel processing
    with Pool(PARALLEL_LEVEL) as p:
        results = p.map(create_random_medium_from_cobra, strains_name)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n=== Processing Summary ===")
    for result in results:
        print(result)
    print(f"\nCheck individual log files in {os.path.abspath(log_dir)}/ for detailed progress")


if __name__ == '__main__':
    main()
