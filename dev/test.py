from edges import EdgeLCIA
import bw2data

# bw2data.projects.set_current("bw25_ei310")
bw2data.projects.set_current("ecoinvent-3.10-cutoff")

act = bw2data.Database("ecoinvent-3.10.1-cutoff").random()
print(act)

# method = ("AWARE 1.2c", "Country", "non", "irri", "yearly")
method = ("GeoPolRisk", "paired", "2024")

LCA = EdgeLCIA({act: 1}, method)
LCA.lci()
LCA.lcia()
print(LCA.score)
