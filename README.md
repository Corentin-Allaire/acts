# A Common Tracking Software (Acts) Project

## Introduction

This project is supposed to be an experiment-independent set of track reconstruction tools. The main philosophy is to provide high-level track reconstruction modules that can be used for any tracking detector. The description of the tracking detector's geometry is optimized for efficient navigation and quick extrapolation of tracks. Converters for several common geometry description languages exist. Having a highly performant, yet largely customizable implementation of track reconstruction algorithms was a primary objective for the design of this toolset. Additionally, the applicability to real-life HEP experiments played a major role in the development process. Apart from algorithmic code, this project also provides an event data model for the description of track parameters and measurements.

Key features of this project include:
* tracking geometry description which can be constructed from TGeo, DD4Hep, or customized input
* simple and efficient event data model,
* performant and highly flexible algorithms for track propagation and fitting,
* basic seed finding algorithms.

The git repository for the Acts project can be found at https://gitlab.cern.ch/acts/acts-core.git.

To get started, you can have a look at our [Getting Started Guide](getting_started.md)

## Mailing list

In order to receive the latest updates, users of the Acts project are encouraged to subscribe to [acts-users@cern.ch](https://e-groups.cern.ch/e-groups/Egroup.do?egroupName=acts-users). This list provides:
- regular updates on the software,
- access to the Acts JIRA project for bug fixes/feature requests,
- a common place for asking any kind of questions.

If you find a bug, have a feature request, or want to contribute to Acts, please have a look at the [contribution guide](CONTRIBUTING.md).

## License and authors

This project is published under the Mozilla Public License 2.0. Details of
this license can be found in the [LICENSE](LICENSE) file or at
http://mozilla.org/MPL/2.0/.

Contributors to the Acts project are listed in [AUTHORS](AUTHORS).

The Acts project contains a copy of [gcovr](http://gcovr.com) licensed under
the 3-Clause BSD license.

This software contains a copy of the
[JSON for Modern C++](https://github.com/nlohmann/json) library by
Niels Lohmann licensed under the MIT License.

The Acts project is based on the ATLAS tracking software. A list of contributors
to the ATLAS tracking repository can be found
[here](http://acts.web.cern.ch/ACTS/ATLAS_authors.html).
