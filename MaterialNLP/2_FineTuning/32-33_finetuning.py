#%% paquetes
from transformers import AutoModelForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.nn.functional import cross_entropy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
# %% YELP Dataset
# fuente: https://huggingface.co/datasets/yelp_review_full/viewer/yelp_review_full/train?f%5blabel%5d%5bvalue%5d=0

#%%  dataset
yelp_hidden_states = joblib.load('model/yelp_hidden_states.joblib')

#%% Modelo y tokenizador
model_name = 'distilbert-base-uncased'
device = 'cuda'
num_labels = 5
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

#%% Dataset
train_ds = yelp_hidden_states.select(range(0, 800))
eval_ds = yelp_hidden_states.select(range(800, 1000))
print(train_ds[0]['input_ids'].shape)
print(eval_ds[0]['input_ids'].shape)
print(yelp_hidden_states[800]['input_ids'].shape)

#%% DatasetDict
yelp_ds_dict = DatasetDict({'train': train_ds, 'test':eval_ds})


#%% argumentos del entrenador
batch_size = 8  # adaptar BS para que quede en la memoria
training_args = TrainingArguments(
    output_dir='./results',          # directorio de salida
    learning_rate=2e-5,              # tasa de aprendizaje
    num_train_epochs=2,              # número total de épocas de entrenamiento
    per_device_train_batch_size=batch_size,  # tamaño de lote por dispositivo durante el entrenamiento
    per_device_eval_batch_size=batch_size,   # tamaño de lote para evaluación
    warmup_steps=500,                # número de pasos de preparación para el programador de tasas de aprendizaje
    disable_tqdm=False,
    push_to_hub=False,
    save_strategy='epoch',
    log_level='error',
    weight_decay=0.01,               # fuerza de caída de peso
    logging_dir='./logs',            # directorio para almacenar registros
    # fuerza de caída de peso
)
#%% entrenador
trainer = Trainer(model=model, 
                  args=training_args, 
                  train_dataset=yelp_ds_dict['train'], eval_dataset=yelp_ds_dict['test'])
trainer.train()

# %% get losses
trainer.evaluate()

# %% calculate predictions
preds = trainer.predict(yelp_ds_dict['test'])

# %%
preds.metrics
# %%
np.argmax(preds.predictions, axis=1)
# %%

# %% confusion matrix
true_classes = yelp_ds_dict['test']['label']
preds_classes = np.argmax(preds.predictions, axis=1)
conf_mat = confusion_matrix(true_classes, preds_classes)
sns.heatmap(conf_mat, annot=True)
# %% accuracy
accuracy_score(true_classes, preds_classes)
# %% entrenamiento del clasificador de referencia

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(yelp_ds_dict['train']['label'], yelp_ds_dict['train']['label'])

# %% precisión del clasificador de referencia
dummy_clf.score(yelp_ds_dict['test']['label'], yelp_ds_dict['test']['label'])

# %% inspeccionar reseñas individuales
model_cpu = model.to('cpu')
#%% inferencia
with torch.no_grad():
    outputs = model_cpu(yelp_ds_dict['test']['input_ids'], yelp_ds_dict['test']['attention_mask'])
#%% cálculo de pérdidas
pred_labels = torch.argmax(outputs.logits, dim=1)
loss = cross_entropy(outputs.logits, yelp_ds_dict['test']['label'], reduction='none')

# %%
df_individual_reviews = pd.DataFrame({'text': yelp_ds_dict['test']['text'], 'label': yelp_ds_dict['test']['label'], 'pred_label': pred_labels, 'loss': loss}).sort_values('loss', ascending=False).reset_index(drop=True)
# %%
df_individual_reviews
# %%
sns.lineplot(data=df_individual_reviews, x='label', y='loss')
# %% guardar el modelo
# login via Terminal: huggingface-cli login
# crear token una una cuenta de HuggingFace


#%%
# %% Push al modelo a HuggingFace Hub
trainer.create_model_card(model_name = 'distilbert-base-uncased-yelp')
trainer.push_to_hub(commit_message='Yelp review classification')

# %% cargar modelo desde HuggingFace Hub
# el nombre fue cambiado en línea a distilbert-base-uncased-yelp
from transformers import pipeline
model_id = 'BertGollnick/distilbert-base-uncased-yelp-new'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
classifier = pipeline('sentiment-analysis', model=model_id, tokenizer=tokenizer)
# %%
# %% visualizar puntuaciones
res = classifier('it is not so great', return_all_scores=True)[0]
df_res = pd.DataFrame(res)
sns.barplot(data=df_res, x='label', y='score')
# %%
