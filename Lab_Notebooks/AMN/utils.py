import platform
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd


def subprocess_ext(command, capture_output=True, text=True, check=True, stream_output=True, **kwargs):
    """
    Execute a command, with automatic WSL wrapping on Windows.

    Args:
        command: Shell command to execute
        capture_output: If True, capture stdout and stderr (for return value)
        text: If True, decode output as text (default True)
        check: If True, raise CalledProcessError on non-zero exit (but show output first)
        stream_output: If True, print output in real-time as it's generated (default True)
        **kwargs: Additional arguments passed to subprocess

    Returns:
        CompletedProcess instance with returncode, stdout, stderr attributes
    """
    if platform.system() == "Windows":
        command = ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-c", command]
        use_shell = False
    else:
        use_shell = True

    if stream_output:
        # Stream output in real-time using Popen
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for simpler handling
            text=text,
            bufsize=1,  # Line buffered
            shell=use_shell,
            **kwargs
        )

        stdout_lines = []

        # Read and print output line by line in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
                if capture_output:
                    stdout_lines.append(line)

        returncode = process.wait()
        stdout_full = ''.join(stdout_lines) if capture_output else None

        # Create result object similar to subprocess.run
        class Result:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        result = Result(returncode, stdout_full, None)  # stderr is merged into stdout

    else:
        # Use original subprocess.run approach (capture then print)
        result = subprocess.run(command, capture_output=capture_output, text=text, check=False, shell=use_shell, **kwargs)

        # Print output if captured
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    # Check for errors after showing output
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr
        )

    return result




def load_experiment_data(per_strain: bool = True,
                         replication: str = "replicates",
                         strain: str = "purB",
                         experiment: str = "mediabotJLF1",
                         well_column: str = "wells",
                         gr_column: str = "mv_mu_max",
                         od_cv_mean_threshold: float = 0.0,
                         od_cv_max_threshold: float = 0.0,
                         od_std_max_threshold: float = 0.0,
                         datasource_path:str = "H:/ROBOT_SCIENTIST/E_coli/Growth_rates/2025-10-31-27/processed"
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load experimental data, growth rates, and statistics; filter based on OD variability.
    """
    flg_has_statistic = True
    if per_strain:
        exp_data_path = f"{datasource_path}/{replication}_STRAINS/{strain}/AMN_dataset/"
        growth_data_path = f"{datasource_path}/{replication}_STRAINS/{strain}/"
        std_data_path = f"{datasource_path}/{replication}/"
        directories = [d for d in Path(std_data_path).iterdir()]
        std_data_list = [ pd.read_csv(d / "predictions.csv") for d in directories if (d / "predictions.csv").exists() ]
        for df , d in zip(std_data_list, directories):
            if "od600_mean" not in df.columns:
                flg_has_statistic = False
                break
            df["experiment"] = d.name
            # Coefficient of Variation (CV), also known as Relative Standard Deviation (RSD).
            df["od_CV"] = df['od600_std'] / df['od600_mean']

        if flg_has_statistic:
            std_data = pd.concat(std_data_list, ignore_index=True)
            std_data = std_data.groupby([well_column,"experiment"]).agg(
                od_mean_max = ( 'od600_mean', 'max' ),
                od_mean_mean = ( 'od600_mean', 'mean' ),
                od_std_max = ( 'od600_std', 'max' ),
                od_std_mean = ( 'od600_std', 'mean' ),
                od_cv_max = ('od_CV', 'max'),
                od_cv_mean = ('od_CV', 'mean'),
            ).reset_index()

    else:
        exp_data_path = f"{datasource_path}/{replication}/{experiment}/AMN_dataset/"
        growth_data_path = f"{datasource_path}/{replication}/{experiment}/"

        std_data = pd.read_csv(growth_data_path + "predictions.csv")
        if "od600_mean" not in std_data.columns:
            flg_has_statistic = False
            std_data = std_data.groupby(well_column).agg(
                od_mean_max = ( 'od600_mean', 'max' ),
                od_mean_mean = ( 'od600_mean', 'mean' ),
                od_std_max = ( 'od600_std', 'max' ),
                od_std_mean = ( 'od600_std', 'mean' ),
            ).reset_index()
    #
    exp_data = pd.read_csv(exp_data_path + "df_flux.csv")
    growth_data = pd.read_csv(growth_data_path + "growth_rates.csv")
    df_levels = pd.read_csv(exp_data_path + "df_AMN_level.csv")

    # Filter growth_data to match the rows in exp_data
    # Since exp_data was created from growth_data where success=='ok', we need to apply the same filter
    # OR simply take the first len(exp_data) rows if they're already aligned
    growth_data = growth_data.reset_index(drop=True)
    growth_data = growth_data.loc[growth_data['success'], :]
    exp_data = exp_data.reset_index(drop=True)
    # Remove growth rate column from growth_data to avoid duplicates
    growth_data = growth_data.drop(columns=[gr_column])
    # Now join them - they should have the same number of rows
    combinded_data = pd.concat([exp_data.reset_index(drop=True), growth_data.reset_index(drop=True)],
                                axis=1)

    if flg_has_statistic:
        if per_strain:
            combinded_data = combinded_data.merge(std_data, on=["experiment", well_column], how="left")
        else:
            combinded_data = combinded_data.merge(std_data, on=[well_column], how="left")

    if flg_has_statistic and od_cv_mean_threshold > 0.0:
        combinded_data = combinded_data.loc[combinded_data["od_cv_mean"] <= od_cv_mean_threshold, :].copy().reset_index(drop=True)
        exp_data = combinded_data[exp_data.columns]
        if combinded_data.shape[0] < 2:
            print("Not enough data.")
            assert False
    elif flg_has_statistic and od_cv_max_threshold > 0.0:
        combinded_data = combinded_data.loc[combinded_data["od_cv_max"] <= od_cv_max_threshold, :].copy().reset_index(drop=True)
        exp_data = combinded_data[exp_data.columns]
        if combinded_data.shape[0] < 2:
            print("Not enough data.")
            assert False
    elif flg_has_statistic and od_std_max_threshold > 0.0:
        combinded_data = combinded_data.loc[combinded_data["od_std_max"] <= od_std_max_threshold, :].copy().reset_index(drop=True)
        exp_data = combinded_data[exp_data.columns]
        if combinded_data.shape[0] < 2:
            print("Not enough data.")
            assert False


    def update_uracil(val):
        for part in ["_0.5", "_22.4", "_200", "_2", "_8", "_64", "_640", "0"]:
                val = val.replace(part, "")
        return val
    combinded_data["supplements_unified"] = combinded_data["supplements"].apply(lambda x: update_uracil(str(x)) if pd.notna(x) else x)

    def classify(val):
        """Classify supplements into Sugar, Nucleo, Amino based on content"""
        if pd.isna(val) or str(val).lower() == 'nan' or str(val).strip() == '':
            return 'None'

        # Split by semicolon and convert to lowercase for comparison
        parts = str(val).lower().split(';')
        parts = [p.strip() for p in parts if p.strip()]

        if not parts:
            print(val)
            return 'None'

        # Define categories (all lowercase)
        sugars = ['glucose', 'succinate', 'sucrose', 'galactose', 'fructose', 'mannose',
                    'maltose', 'lactose', 'xylose', 'arabinose', 'ribose']

        nucleobases = ['adenine', 'uracil', 'guanine', 'cytosine', 'thymine']

        # Amino acids are typically 3 letters (all lowercase)
        amino_acids_3letter = ['ala', 'arg', 'asn', 'asp', 'cys', 'gln', 'glu', 'gly',
                                'his', 'ile', 'leu', 'lys', 'met', 'phe', 'pro', 'ser',
                                'thr', 'trp', 'tyr', 'val']

        categories = []
        for part in parts:
            # Check if it's a sugar
            if any(sugar in part for sugar in sugars):
                if 'Sugar' not in categories:
                    categories.append('Sugar')
            # Check if it's a nucleobase
            elif any(nucleo in part for nucleo in nucleobases):
                if 'Nucleo' not in categories:
                    categories.append('Nucleo')
            # Check if it's a 3-letter amino acid or contains common amino acid patterns
            elif any(aa in part for aa in amino_acids_3letter) or len(part) == 3:
                if 'Amino' not in categories:
                    categories.append('Amino')

        if not categories:
            return 'Other'

        # Sort to ensure consistent ordering
        categories.sort()
        return '+'.join(categories)

    combinded_data["group"] = combinded_data["supplements_unified"].apply(classify)

    return combinded_data, exp_data, df_levels
