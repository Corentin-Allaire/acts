# This file read the data from hits.csv file and tries to convert the geometry id using the bit map

import pandas as pd
import numpy as np
import glob
import sys

# Define the masks as constants
kVolumeMask = 0xFF00000000000000
kBoundaryMask = 0x00FF000000000000
kLayerMask = 0x0000FFF000000000
kApproachMask = 0x0000000FF0000000
kSensitiveMask = 0x000000000FFFFF00
kExtraMask = 0x00000000000000FF


def extract_masked_values(value):
    """
    Function to extract masked values from an input value
    Args:
        - value (int): The input value
    Returns:
        Tuple[int]: The extracted values of the volume, layer, sensitive and extra Ids
    """
    volume = (value & kVolumeMask) >> 56
    layer = (value & kLayerMask) >> 36
    sensitive = (value & kSensitiveMask) >> 8
    extra = value & kExtraMask
    return [volume, layer, sensitive, extra]


def main():
    """
    Main function to run the preprocessing of the data. In this preprocessing, the data is read from the different files (on per event)
    and the particles are filtered to keep only the ones with at least 7 hits in the data and ordered by decreasing pT. 
    Hits outside the seeing region are removed and the geometry id is converted to volume, layer, sensitive and extra values. 
    Finally the data is saved in a new single file.
    """

    # Read the data from all the odd_output/event*-hits.csv file and the odd_output/event*-particles.csv file
    # hit_files = sorted(glob.glob("odd_output" + "/event*-hits.csv"))
    hit_files = sorted(glob.glob("odd_output" + "/event000000000-hits.csv"))
    # particle_files = sorted(glob.glob("odd_output" + "/event*-particles.csv"))

    # particle_files = sorted(glob.glob("odd_output" + "/event*-particles_initial.csv"))
    particle_files = sorted(
        glob.glob("odd_output" + "/event000000000-particles_initial.csv")
    )

    # Loop through all the files
    full_data = pd.DataFrame()
    full_particles = pd.DataFrame()
    counter = 0
    for file, particle_file in zip(hit_files, particle_files):
        # Every 100 events, print the current event number
        if counter % 100 == 0:
            print("Processing event ", counter)
        data = pd.read_csv(file)
        particles = pd.read_csv(particle_file)
        # loop over all particles and remove the ones where the ids appear less than 7 times in the data
        for index, row in particles.iterrows():
            particle_id = row["particle_id"]
            if len(data[data["particle_id"] == particle_id]) < 7:
                particles = particles[particles["particle_id"] != particle_id]

        # Remove the particles with no momentum
        particles = particles[particles["pz"] != 0] 

        data = data.drop(columns=["tpx", "tpy", "tpz", "te", "deltapx", "deltapy", "deltapz", "deltae", "index"])
        # Remove all entry with |tz|> 3000 and sqrt(tx^2 + ty^2) > 250
        data = data[(data['tz'] < 3000) & (data['tz'] > -3000) & (data['tx']**2 + data['ty']**2 < 200**2)]
        # Compute the value of eta (pseudo rapidity) and phi (angle with respect to the the z axis) for all entry base on a starting position at (0,0,0):
        data["eta"] = -1 * np.log(np.tan(np.arctan2(np.sqrt(data["tx"] ** 2 + data["ty"] ** 2), data["tz"]) / 2))
        data["phi"] = np.arctan2(data["ty"], data["tx"])
        data["r"] = np.sqrt(data["tx"] ** 2 + data["ty"] ** 2)
        # Same for particles using the momentum
        particles["eta"] = -1 * np.log(np.tan(np.arctan2(np.sqrt(particles["px"] ** 2 + particles["py"] ** 2),particles["pz"]) / 2))
        particles["phi"] = np.arctan2(particles["py"], particles["px"])
        particles["pT"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
        # Sort the particles by decreasing pT
        particles = particles.sort_values("pT", ascending=False)
        # Use the bit map to extract the volume, layer, sensitive and extra values
        data["volume"], data["layer"], data["sensitive"], data["extra"] = zip(*data["geometry_id"].map(extract_masked_values))
        data = data.drop(columns=["geometry_id"])

        # Add an event id to the data
        data["event_id"] = counter
        particles["event_id"] = counter
        counter += 1
        full_data = pd.concat([full_data, data])
        full_particles = pd.concat([full_particles, particles])

    # Write the new files
    full_data.to_csv("odd_output/hits.csv", index=False)
    full_particles.to_csv("odd_output/particles.csv", index=False)

if __name__ == "__main__":
    sys.exit(main())
