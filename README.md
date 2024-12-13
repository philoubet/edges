# ``edges``: Edge-based life cycle impact assessment

[![PyPI version](https://badge.fury.io/py/edges.svg)](https://badge.fury.io/py/csc-brightway)

``edges`` is a Python Library for Contextual Impact Assessment in 
Life Cycle Analysis (LCA) for the ``brightway2`` LCA framework.

The *edges* Python library introduces an innovative approach to the application 
of characterization factors during the impact assessment phase of Life Cycle 
Assessment (LCA). Unlike traditional impact assessment methods that apply 
characterization factors to *nodes* (e.g., *nodes* like `Carbon dioxide, fossil` in the 
``brightway2`` ecosystem, which is assigned a Global Warming Potential (GWP) 
factor of 1), ``edges`` applies these factors directly to *edges* between *nodes*.
To do this, ``edges`` injects off-diagonal values in the *characterization matrix*, of
which the value is determined by the context of the edge.

*Edges* represent the relationships or exchanges between *nodes*, allowing ``edges`` 
to leverage contextual information such as the attributes of both suppliers 
and consumers. This approach enables a more nuanced and flexible characterization 
of impacts, incorporating additional parameters such as the geographic location 
of the consumer or the magnitude of the exchanged flow. For instance, the 
library can adjust the characterization factor based on the location of the 
consumer or scale it relative to the amount exchanged, offering a more 
context-sensitive impact assessment framework.

## Features

- **Country-specific characterization factors** for water-related impacts.
- **Seamless integration** with the Brightway LCA framework.
- Implements national and sub-national characterization factors of the **AWARE method 1.2c**.
- Future updates will include additional impact categories.

## Installation

You can install the library using pip:

```bash
pip install git+https://github.com/romainsacchi/edges.git
```

## Getting Started

### Check available methods from ``edges``

```python
    
from edges import get_available_methods

# Get the available methods
methods = get_available_methods()
print(methods)

```

### Perform edge-based LCIA with ``edges``

```python
import bw2data
from edges import SpatialLCA

# Select an activity from the LCA database
act = bw2data.Database("ecoinvent-3.10-cutoff").random()

# Define a method
method = ('AWARE 1.2c', 'Country', 'unspecified', 'yearly')

# Initialize the LCA object
LCA = SpatialLCA({act: 1}, method)
LCA.lci()
# Perform the LCAI calculation
LCA.lcia()
LCA.score

# Print a dataframe with the characterization factors used
LCA.generate_cf_table()

```

## Data Sources

* **AWARE**: The AWARE factors are adapted from peer-reviewed sources and tailored to provide 
precise country-specific data for environmental modeling. Refer to the AWARE 
website [https://wulca-waterlca.org/](https://wulca-waterlca.org/) for more information.

## Methodology

1. ``edges`` introduces edge-specific characterization factors
in the characterization matrix of ``bw2calc`` before performing the LCA calculation.
The characterization factors are stored in the ``data`` folder of the library. 
Currently, ``edges`` provides characterization factors for 346 national and 
sub-national regions, based on the [AWARE](https://wulca-waterlca.org/aware/) method,
based on the location of edge consumers.

2. For specific ``ecoinvent`` regions (e.g., RER, Canada without Quebec, etc.), 
``edges`` computes the weighted average of the characterization factors for the 
countries included in the region, based either on population or GDP. The weighting 
key can be selected by the user (weighting by population size by default).

3. For relative regions (e.g., RoW, RoE, etc.), ``edges`` dynamically defines the 
locations included in the region based on the activities in the LCA database. 
The weighted average of the characterization factors is then computed based on the 
activities' locations.

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. **Fork** the repository.
2. **Create** a new branch for your feature or fix.
3. **Commit** your changes.
4. **Submit** a pull request.


## License
This project is licensed under the MIT License.
See the [LICENSE.md](LICENSE.md) file for more information.

## Contact
For any questions or inquiries, please contact the project maintainer 
at [romain.sacchi@psi.ch](mailto:romain.sacchi@psi.ch).

## Contributors

- [Romain Sacchi](https://github.com/romainsacchi)
- [Alvaro Hahn Menacho](https://github.com/alvarojhahn)
- [Joanna Schlesinger](https://github.com/joanna-schles)
- [Juliana Steinbach](https://github.com/juliana-steinbach)
- Thomas Beaussier

## Acknowledgments
The development of this library was entirely supported by the French agency for 
Energy [ADEME](https://www.ademe.fr/), via the financing of the [HySPI](https://www.isige.minesparis.psl.eu/actualite/le-projet-hyspi/) project.
The HySPI project aims to provide a methodological framework to analyze and 
quantify, in a systemic and prospective manner, the environmental impacts of the 
decarbonization strategy of H2 production used by the industry in France.
