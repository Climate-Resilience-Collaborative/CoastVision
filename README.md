# CoastVision
[![Last Commit](https://img.shields.io/github/last-commit/Climate-Resilience-Collaborative/CoastVision)](
https://github.com/Climate-Resilience-Collaborative/CoastVision/commits/)
![GitHub issues](https://img.shields.io/github/issues/Climate-Resilience-Collaborative/CoastVision)
[![GitHub release](https://img.shields.io/github/release/Climate-Resilience-Collaborative/CoastVision)](https://GitHub.com/Climate-Resilience-Collaborative/CoastVision/releases/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/Climate-Resilience-Collaborative/CoastVision)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2779293.svg)](https://doi.org/10.5281/zenodo.2779293)
[![Join the chat at https://gitter.im/CoastSat/community](https://badges.gitter.im/spyder-ide/spyder.svg)](https://gitter.im/CoastSat/community)

CoastVision is an open-source Python framework geared towards generating satellite-derived shorelines (SDS) in [PlanetScope](https://developers.planet.com/docs/data/planetscope/) imagery. Given a time window and an area of interest (AOI) CoastVision will:

<details>
<summary><strong>Download applicable PlanetScope imagery</strong></summary>
There are interesting things in here
</details>
<details>
<summary><strong>Co-register Imagery using AROSICS</strong></summary>
<a href="https://danschef.git-pages.gfz-potsdam.de/arosics/doc/">AROSICS</a> is an open-source <a href="https://pypi.org/project/arosics/">Python package</a> which performs image co-registration for multi-sensor satellite data.
</details>
<details>
<summary><strong>Segment image into land and water</strong></summary>
</details>
<details>
<summary><strong>Extract land/water interface</strong></summary>
</details>


### Table of Contents

- [Installation](#installation)
- [Usage](#usage)
   - [Retrieval of the satellite images in GEE](#retrieval)
- [References and Datasets](#references)


## 1. Installation<a name="introduction"></a>
Use `coastvision.yml` to create conda environment. This will take a few minutes.
```
cd path/to/CoastVision
conda env create -f coastvision.yml
conda activate coastvision
```
