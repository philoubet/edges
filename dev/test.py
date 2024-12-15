from edges import EdgeLCIA
import bw2data

bw2data.projects.set_current("bw25_ei310")

act = bw2data.Database("ecoinvent-3.10-cutoff").random()

method = ('AWARE 1.2c', 'Country', 'non', 'irri', 'yearly')

LCA = EdgeLCIA({act: 1}, method)
LCA.lci()
LCA.lcia()
print(LCA.score)
