import glob

import pandas as pd
<<<<<<< HEAD
import numpy as np


def matchGood(Seed_files: list[str], CKF_files: list[str]) -> pd.DataFrame:
    """Read the dataset from the different files, remove the particle with only fakes and combine the datasets"""
    """
    @param[in] Seed_files: DataFrame contain the data from each seed files (1 file per events usually)
    @return: combined DataFrame containing all the seed, ordered by events and then by truth particle ID in each events 
=======


def matchGood(seed_files: list[str], ckf_files: list[str]):
    """Read the dataset from the tracks and seeds files, then modify the seed dataset so that good seed correspond to the ones that lead to good tracks. Seed with truth id that do not lead to good tracks are considered as fake. Also create a new dataset with only truth particle associated to a good seeds."""
    """
    @param[in] Seed_files: List of files containing seeds data (1 file per events usually)
    @param[in] CKF_files: List of files containing tracks data (1 file per events usually)
>>>>>>> upstream/main
    """
    data_seed = pd.DataFrame()
    data_track = pd.DataFrame()
    goodSeed = pd.DataFrame()
<<<<<<< HEAD
    data = pd.DataFrame()
    # Loop over the different track files and collect the list of seed ID associated to the good tracks
    for f_ckf, f_seed in zip(CKF_files, Seed_files):
=======
    # Loop over the different track files and collect the list of seed ID associated to the good tracks
    for f_ckf, f_seed in zip(ckf_files, seed_files):
>>>>>>> upstream/main
        print("reading file: ", f_ckf, f_seed)
        data_track = pd.read_csv(f_ckf)
        data_track = data_track.loc[data_track["good/duplicate/fake"] == "good"]
        goodSeed = data_track["seed_id"]

        data_seed = pd.read_csv(f_seed)
        # Add a good seed column to the seed dataset
        data_seed["goodSeed"] = data_seed["seed_id"].isin(goodSeed)

<<<<<<< HEAD
        data_seed.loc[data_seed["good/duplicate/fake"] == "good", "good/duplicate/fake"] = "duplicate"
=======
        data_seed.loc[
            data_seed["good/duplicate/fake"] == "good", "good/duplicate/fake"
        ] = "duplicate"
>>>>>>> upstream/main
        data_seed.loc[data_seed["goodSeed"] == True, "good/duplicate/fake"] = "good"

        cleanedData = pd.DataFrame()

<<<<<<< HEAD
=======
        # Find the particle ID that are associated to only fake seeds
>>>>>>> upstream/main
        for ID in data_seed["particleId"].unique():
            if (
                data_seed.loc[data_seed["particleId"] == ID, "goodSeed"] == False
            ).all():
<<<<<<< HEAD
                data_seed.loc[data_seed["particleId"] == ID, "good/duplicate/fake"] = "fake"
            else:
                cleanedData = pd.concat([data_seed.loc[data_seed["particleId"] == ID], cleanedData])

        # Save the cleaned dataset for future use (the cleaning is time consuming)
=======
                data_seed.loc[data_seed["particleId"] == ID, "good/duplicate/fake"] = (
                    "fake"
                )
            else:
                cleanedData = pd.concat(
                    [data_seed.loc[data_seed["particleId"] == ID], cleanedData]
                )

        # Save the matched dataset for future use (the matching is time consuming)
>>>>>>> upstream/main
        matched = f_seed[:-4] + "_matched.csv"
        matchedData = data_seed.sort_values("seed_id")
        matchedData = matchedData.set_index("seed_id")
        matchedData = matchedData.drop(columns=["goodSeed"])
        matchedData.to_csv(matched)
<<<<<<< HEAD
        data = pd.concat([data, matchedData])
=======
>>>>>>> upstream/main

        # Save the cleaned dataset for future use (the cleaning is time consuming)
        cleaned = f_seed[:-4] + "_cleaned.csv"
        cleanedData = cleanedData.sort_values("seed_id")
        cleanedData = cleanedData.set_index("seed_id")
        cleanedData = cleanedData.drop(columns=["goodSeed"])
        cleanedData.to_csv(cleaned)
<<<<<<< HEAD
        
    return data


# ttbar events used as the training input
seed_files = sorted(glob.glob("odd_output" + "/event*-seed.csv"))
CKF_files = sorted(glob.glob("odd_output" + "/event*-tracks_ckf.csv"))
data = matchGood(seed_files, CKF_files)
=======

    return


# Read the seed and track files and match them
# This will allow us to determine which seeds leads to the best possible tracks
seed_files = sorted(glob.glob("odd_output" + "/event*-seed.csv"))
ckf_files = sorted(glob.glob("odd_output" + "/event*-tracks_ckf.csv"))
matchGood(seed_files, ckf_files)
>>>>>>> upstream/main
