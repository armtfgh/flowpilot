#%%
import pyalex
import requests



pyalex.config.api_key = "nIjZIvaAH18StQO3WHVed5"


# %%
r = pyalex.Works().search("Buchwald Hartwig continuous flow").filter(
    has_abstract=True
).sort(cited_by_count="desc").get(per_page=5)
# %%
