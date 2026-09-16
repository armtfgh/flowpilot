
#%%
#this is when new pdf os added to the records
from visualization.llm_classifier import classify_all
clf = classify_all(verbose=True)

# %%
from visualization.fig1a_reaction_classes import make_figure as fig1a
from visualization.fig1h_knowledge_graph import make_figure as fig1h
from visualization.fig1b_reactor_types import make_figure as fig1b
from visualization.fig1c_reactor_materials import make_figure as fig1c
from visualization.fig1d_bond_types import make_figure as fig1d
from visualization.fig1e_light_sources import make_figure as fig1e
from visualization.fig1f_streams import make_figure as fig1f
from visualization.fig1g_yield_distribution import make_figure as fig1g 

fig = fig1a()
fig.savefig("fig1a.png", dpi=300, bbox_inches="tight")

fig = fig1b()
fig.savefig("fig1b.png", dpi=300, bbox_inches="tight")

fig = fig1c()
fig.savefig("fig1c.png", dpi=300, bbox_inches="tight")

fig = fig1d()
fig.savefig("fig1d.png", dpi=300, bbox_inches="tight")

fig = fig1e()
fig.savefig("fig1e.png", dpi=300, bbox_inches="tight")

fig = fig1f()
fig.savefig("fig1f.png", dpi=300, bbox_inches="tight")

fig = fig1g()
fig.savefig("fig1g.png", dpi=300, bbox_inches="tight")






# %%
import os                                                                                                                                                                                                                                                                                                                                                                                                                             

from visualization.rule_classifier import classify_all_rules                                                                                                                                                                                
classify_all_rules()   # ~$0.03, ~2-3 minutes for 2537 rules


# %%
from visualization.fig2a_rule_landscape import make_figure as fig2a
fig = fig2a()
fig.savefig("fig2a.png", dpi=300, bbox_inches="tight")

from visualization.fig2b_formula_coverage import make_figure as fig2b
fig = fig2b()
fig.savefig("fig2b.png", dpi=300, bbox_inches="tight")

from visualization.fig2c_coverage_heatmap import make_figure as fig2c
fig = fig2c()
fig.savefig("fig2c.png", dpi=300, bbox_inches="tight")

from visualization.fig2d_concept_network import make_figure as fig2d
fig = fig2d()
fig.savefig("fig2d.png", dpi=300, bbox_inches="tight")



# %% 

from visualization.fig3a_rag_architecture import make_figure                                                                                                                                                                                
from visualization.fig3b_query_enrichment import make_figure as f3b                                                                                                                                                                         
from visualization.fig3c_score_decomposition import make_figure as f3c  
from visualization.fig3d_rag_quality import make_figure as f3d

                                                                                                                                                                                                                                            
fig3a = make_figure()                                                                                                                                                                                                                       
fig3b = f3b()                                                                                                                                                                                                                               
fig3c = f3c(n_queries=80)  # increase to 437 for full dataset
fig3d = f3d()




# %%
