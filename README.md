# ``edges``: Edge-based life cycle impact assessment

[![PyPI version](https://badge.fury.io/py/edges.svg)](https://badge.fury.io/py/csc-brightway)

``edges`` is a library allowing flexible Life Cycle Impact Assessment (LCIA) 
for the ``brightway2``/``brightway25`` LCA framework.

Unlike traditional LCIA methods that apply characterization factors (CFs) solely to `nodes` 
(e.g., elementary flows), `edges` applies CFs directly on the edges — the exchanges between 
suppliers and consumers — allowing for more precise and context-sensitive impact characterization.

This approach enables LCIA factors to reflect the specific context of each exchange, including parameters such as:

* Geographic region of production and consumption 
* Magnitude of flows 
* Scenario-based parameters (e.g., changing atmospheric conditions)

The ``edges`` Python library offers a novel approach to applying characterization factors 
(CFs) during the impact assessment phase of Life Cycle Assessment (LCA). 
Unlike conventional methods that uniformly assign CFs to *nodes* (e.g., processes 
like ``Water, from well`` in the brightway2 ecosystem), ``edges`` shifts the focus to the 
*edges*—the *exchanges* or relationships between *nodes*. This allows CFs to be conditioned 
based on the specific context of each *exchange*. Essentially, ``edges`` introduces unique 
values in the characterization matrix tailored to the characteristics of each *edge*.

By focusing on *edges*, the library incorporates contextual information such as the 
attributes of both the *supplier* and the *consumer* (e.g., geographic location, ISIC 
classification, amount exchanged, etc.). This enables a more detailed and flexible 
impact characterization, accommodating parameters like the location of the consumer 
and the magnitude of the exchange.

Furthermore, ``edges`` supports the calculation of weighted CFs for both static regions 
(e.g., RER) and dynamic regions (e.g., RoW), enhancing its ability to model complex 
and region-specific scenarios.

## Key Features

* Edge-based CFs: Assign CFs specifically to individual exchanges between processes. 
* Geographic resolution: Supports 346 national and sub-national regions. 
* Scenario-based flexibility: Incorporate parameters (e.g., CO₂ atmospheric concentration) directly in CF calculations, enabling dynamic scenario analysis. 
* Efficient workflow: Clearly separates expensive exchange-mapping tasks (performed once) from inexpensive scenario-based numeric CF evaluations.

Currently, the library provides regionalized CFs for:

* AWARE 1.2c (water scarcity impacts)
* ImpactWorld+ 2.1
* GeoPolRisk 1.0

## Installation

You can install the library using pip:

```bash
pip install git+https://github.com/Laboratory-for-Energy-Systems-Analysis/edges.git
```

## Getting Started

Check out the [examples' notebook](https://github.com/romainsacchi/edges/blob/main/examples/examples.ipynb).

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
from edges import EdgeLCIA

# Select an activity from the LCA database
act = bw2data.Database("ecoinvent-3.10-cutoff").random()

# Define a method
method = ('AWARE 1.2c', 'Country', 'unspecified', 'yearly')

# Initialize the LCA object
LCA = EdgeLCIA({act: 1}, method)
LCA.lci()
# Map CFs to exchanges
LCA.map_exchanges()
# If needed, extend the mapping to aggregated and `dynamic` regions (e.g., RoW)
LCA.map_aggregate_locations()
LCA.map_dynamic_locations()
LCA.map_disaggregate_locations()
# add global CFs to exchanges missing a CF
LCA.map_remaining_locations_to_global()
# Evaluate CFs
LCA.evaluate_cfs()
# Perform the LCIA calculation
LCA.lcia()
print(LCA.score)

# Print a dataframe with the characterization factors used
LCA.generate_cf_table()

```

### Perform parameter-based LCIA

Consider the following LCIA data file (saved under `gwp_example.json`)`:

```json
[
  {
    "supplier": {
      "name": "Carbon dioxide",
      "operator": "startswith",
      "matrix": "biosphere"
    },
    "consumer": {
      "matrix": "technosphere",
      "type": "process"
    },
    "value": "1.0"
  },
  {
    "supplier": {
      "name": "Methane, fossil",
      "operator": "contains",
      "matrix": "biosphere"
    },
    "consumer": {
      "matrix": "technosphere",
      "type": "process"
    },
    "value": "28 * (1 + 0.001 * (co2ppm - 410))"
  },
  {
    "supplier": {
      "name": "Dinitrogen monoxide",
      "operator": "equals",
      "matrix": "biosphere"
    },
    "consumer": {
      "matrix": "technosphere",
      "type": "process"
    },
    "value": "265 * (1 + 0.0005 * (co2ppm - 410))"
  }
]

```

We can perform a parameter-based LCIA calculation as follows:

```python

import bw2data
from edges import EdgeLCIA

# Select an activity from the LCA database
act = bw2data.Database("ecoinvent-3.10-cutoff").random()

# Define scenario parameters (e.g., atmospheric CO₂ concentration and time horizon)
params = {"co2ppm": [410, 450, 500], "h": 100}

# Define an LCIA method (symbolic CF expressions stored in JSON)
method = ('GWP', 'scenario-dependent', '100 years')

# Initialize LCIA
lcia = EdgeLCIA(
   demand={act: 1},
   filepath="gwp_example.json",
   parameters=params
)

# Perform inventory calculations (once)
lcia.lci()

# Map exchanges to CF entries (once)
lcia.map_exchanges()

# Optionally, resolve geographic overlaps and disaggregations (once)
lcia.map_aggregate_locations()
lcia.map_dynamic_locations()
lcia.map_remaining_locations_to_global()

# Run scenarios efficiently
results = []
for idx in range(lcia.scenario_length):
    lcia.evaluate_cfs(idx)
    lcia.lcia()
    df = lcia.generate_cf_table()

    scenario_result = {
        "scenario": idx,
        "co2ppm": params["co2ppm"][idx],
        "score": lcia.score,
        "CF_table": df
    }
    results.append(scenario_result)

    print(f"Scenario {idx+1} (CO₂ {params['co2ppm'][idx]} ppm): Impact = {lcia.score}")
    
```


## Data Sources

* **AWARE**: The AWARE factors are adapted from peer-reviewed sources and tailored to provide 
precise country-specific data for environmental modeling. Refer to the AWARE 
website [https://wulca-waterlca.org/](https://wulca-waterlca.org/) for more information.

If you use the AWARE method, please cite the following publication:

```bibtex
@article{boulay2018aware,
  title={The WULCA consensus characterization model for water scarcity footprints: assessing impacts of water consumption based on available water remaining (AWARE).},
  author={Anne-Marie Boulay, Jane Bare, Lorenzo Benini, Markus Berger, Michael J. Lathuillière, Alessandro Manzardo, Manuele Margni, Masaharu Motoshita, Montserrat Núñez, Amandine Valerie Pastor, Bradley Ridoutt, Taikan Oki, Sebastien Worbe & Stephan Pfister },
  journal={Int J Life Cycle Assess},
  volume={23},
  pages={368–378},
  year={2018},
  publisher={Springer}
}
```

## Methodology

1. **Exchange Mapping**: Identifies the inventory exchanges matching the CF criteria. Matching can be based on any exchange fields (e.g., name, unit, location, classification, etc.).

2. **Scenario Evaluation**: CF values can be defined numerically or symbolically and may depend on parameters (e.g., co2ppm, time horizons). 

3. **Characterization Matrix**: CFs are applied directly to inventory exchanges in the characterization matrix of Brightway, enabling precise context-based LCIA.

Additionally, for regionalized characterization factors, helping functions are available:

* `.map_aggregate_location()`: For specific ``ecoinvent`` regions (e.g., RER, Canada without Quebec, etc.) that do ont have regionalized CFs, 
``edges`` computes the weighted average of the characterization factors for the 
countries included in the region, based either on population or GDP. The weighting 
key can be selected by the user (weighting by population size by default), or it can be provided within the LCIA method file. 

* `.map_dynamic_locations()`: For relative regions (e.g., RoW, RoE, etc.), ``edges`` dynamically defines the 
locations included in the region based on the mathing activities in the LCA database. The weighted average of the characterization factor of the geographies containd in the region is then computed accordingly.

* `.map_disaggregate_locations()`: For locations that do not have regionalized CFs, but are part of larger regions that have a regionalized CF, ``edges`` will attribute such CF to the exchange.

* `.map_remaining_locations_to_global()`: For exchanges that do not have a regionalized CF, ``edges`` will attribute the global CF to the exchange. The global CF is defined as the weighted-average of the regionalized CFs, based on the weighting key selected.


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

## Acknowledgments
The development of this library was entirely supported by the French agency for 
Energy [ADEME](https://www.ademe.fr/), via the financing of the [HySPI](https://www.isige.minesparis.psl.eu/actualite/le-projet-hyspi/) project.
The HySPI project aims to provide a methodological framework to analyze and 
quantify, in a systemic and prospective manner, the environmental impacts of the 
decarbonization strategy of H2 production used by the industry in France.
