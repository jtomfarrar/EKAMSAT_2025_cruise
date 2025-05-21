# EKAMSAT 2025 Cruise Leg 1 (TN-444)

## Cruise Report Abstract

Leg 1 of Voyage TN-444 on the R/V Thomas G. Thompson set sail from Phuket, Thailand on 3 May 2025 to make measurements and deploy oceanographic and meteorological instruments in the central Bay of Bengal as part of the US-India EKAMSAT initiative (Enhancing Knowledge of the Arabian Sea Marine Environment through Science and Advanced Training). The expedition took place in two cruise legs. This cruise report describes the activities of Leg 1, which started and ended in Phuket, Thailand, spanning 3-15 May 2025.

The primary goals of TN-444 Leg 1 were to: (1) identify an area in the pre-monsoon "warm pool" in central Bay of Bengal that could serve as a focal point for the rest of the experiment; (2) deploy 3 Seagliders and 5 Wave Gliders to collect oceanographic and atmospheric measurements; (3) broadly distribute 12-14 wave drifters around the Bay of Bengal; (4) survey the area with the shipboard instruments (ADCP, thermosalinograph, shipboard meteorological instruments, X-band surface current radar, other underway instruments), an Underway CTD system, a ship-mounted ceilometer, and a ship-mounted, motion-corrected LIDAR wind profiler. An ancillary goal, part of the SO-PACE project, was to collect in situ data that can be used to validate PACE hyperspectral satellite products and derive uncertainties.

## Repository Purpose

This repository contains code for generating figures and plots used in the EKAMSAT 2025 cruise report. The scripts in `src/` are used to create:
- Maps of the cruise track
- Plots of large-scale oceanographic conditions, including sea surface temperature (SST) and sea surface height (SSH)
- Visualizations of cruise waypoints and instrument deployment locations

The code is intended to support cruise documentation, reporting, and scientific analysis by providing reproducible scripts for data visualization and mapping relevant to the EKAMSAT 2025 Leg 1 cruise.

## Table of Contents

- [EKAMSAT 2025 Cruise Leg 1 (TN-444)](#ekamsat-2025-cruise-leg-1-tn-444)
  - [Cruise Report Abstract](#cruise-report-abstract)
  - [Repository Purpose](#repository-purpose)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Data Sources](#data-sources)
  - [Results](#results)
  - [Acknowledgments](#acknowledgments)
  - [References](#references)

## Installation

It is recommended to use a conda or mamba environment to manage dependencies for this project. To set up the environment, run:

```bash
conda env create -f environment.yml
# or, with mamba (faster):
mamba env create -f environment.yml
```

Then activate the environment:

```bash
conda activate ekamsat_2025
```

If you need to update the environment after changes to `environment.yml`, use:

```bash
conda env update -f environment.yml --prune
```

## Usage

To generate the figures and plots for the cruise report, run each of the plotting scripts in the `src/` directory individually (except for `functions.py`, which contains shared functions and is not meant to be executed directly). For example:

```bash
python src/download_DUACS.py
python src/plot_BoB_SST_context.py
python src/plot_EKAMSAT2025_UCTD.py
python src/plot_EKAMSAT2025_cruise_track.py
```

Each script will process the relevant data and create output figures and plots in the `output/` directory.


## Results

The results of the data processing and visualization are saved in the `img/` directory. This includes:
- Figures and plots for the cruise report
- Maps showing the cruise track and instrument deployment locations
- Time series plots of oceanographic and atmospheric conditions

## Acknowledgments

We would like to thank the captain and crew of the R/V Thomas G. Thompson for their support during the EKAMSAT 2025 TN-444 cruise. Their expertise and assistance were invaluable in the successful completion of this expedition.

## References

- EKAMSAT Initiative: [https://www.example.com/ekamsat](https://www.example.com/ekamsat)
- R/V Thomas G. Thompson: [https://www.example.com/thomas-g-thompson](https://www.example.com/thomas-g-thompson)

## Contact
For questions about the repository or contributions, please contact:

J. Thomas Farrar
Senior Scientist, Physical Oceanography Department
Woods Hole Oceanographic Institution
Email: jfarrar@whoi.edu