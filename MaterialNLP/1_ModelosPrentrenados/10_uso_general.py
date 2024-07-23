#%% paquetes
from transformers import pipeline

# %% asignamos solo una tarea
pipe = pipeline(task="text-classification")
# %% run pipe
pipe("I like it very much.")
# %% asignamos modelo
pipe = pipeline(task="text-classification", 
model="nlptown/bert-base-multilingual-uncased-sentiment")
# %% run pipe
# consumimos una cadena
pipe("I like it very much.")

# %% consumimos una lista
pipe(["I like it very much.", 
      "I hate it."])


# %%
