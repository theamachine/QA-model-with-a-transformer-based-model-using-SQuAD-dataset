
# Question Answering on SQuAD dataset

<img src='https://miro.medium.com/max/1200/1*6ga054VrBMwLG4OC13bQoQ.png'>

<br />

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.


<br />

# Project Overview 

In this project Stanford question answering dataset was used, also, a question answering model was built with a transformer-based architecture on a generic task and then finetuned on the task at hand. The transformer’s implementation that will was used is provided by HuggingFace library. Some data cleaning techniques were used in preprocessing. Nevertheless, Clustering and classification techniques were applied with various number of models.



# Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install these packages.

we will use a transformer-based architecture.<br>
The transformer used will be pre-trained on a generic task and then finetuned on
the task at hand.<br>
The transformers' implementation that will be used will be provided by
**HuggingFace** library.<br>
Let's start by installing it.

```python
! pip install datasets transformers
```


pyLDAvis is a python library for interactive topic model visualization. This is a port of the fabulous R package by Carson Sievert and Kenny Shirley.

pyLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.

```bash
! pip install pyLDAvis
```




# Usage

```python
from datasets import load_dataset
import pandas as pd
from IPython.display import display, HTML
from datasets import Dataset, DatasetDict
import collections
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import zipfile
from transformers import default_data_collator
from z3 import *
from tqdm.notebook import tqdm
from datasets import load_metric
import re
import string
import pyLDAvis.gensim_models as gensimvis
import nltk
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import scipy.cluster.hierarchy as sch
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pyLDAvis
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
%matplotlib inline



## How To Run this Notebook 
first ,you nead to install necessary packages.
then you can run this notebook secuntially becasue each part depends on the one before it
firest part is transformer-based architecture 
second part is clustering 
then we export the dataset to a csv file  and then we can use it in the next part
third part is classification

we will use a transformer-based architecture.<br>
The transformer used will be pre-trained on a generic task and then finetuned on
the task at hand.<br>
The transformers' implementation that will be used will be provided by
**HuggingFace** library.<br>
Let's start by installing it.

```python
! pip install datasets transformers
```

## Loading the Dataset

### Dataset Downloading
The dataset is a .json file loaded in a google drive.

```python
!gdown --id "1aURk7-EAowXK-KXy7Ut1Y3z1X18kHv0E"
```

### Dataset Creation

The dataset will be loaded using HuggingFace's loading function.

```python
from datasets import load_dataset

json_file_path = "training_set.json"
ds_original = load_dataset('json', data_files= json_file_path, field='data')
```

HuggingFace's loading function returns a dict-link object called `DatasetDict`
that incapsulate the real dataset.
The dataset loaded will be stored under the key "train", as such it will
subsequently splitted according to the projects requirenmets.

```python
ds_original
```

```python
# Print the 1st row
ds_original['train'][0]
```

we need to convert json file to dataframe to facilitaye dealing wuth it.

```python
def generate_dataset(dataset, test = False):
  for data in dataset["train"]:
    title = data.get("title", "").strip()
    for paragraph in data["paragraphs"]:
      context = paragraph["context"].strip()
      for qa in paragraph["qas"]:
          # Handling questions
          question = qa["question"].strip()
          id_ = qa["id"]
          # Answers won't be present in the testing (compute_answers.py)
          if not test:
              # Handling answers
              for answer in qa["answers"]:
                answer_start = [answer["answer_start"]]
              for answer in qa["answers"]:
                answer_text = [answer["text"].strip()]

              yield id_, {
                "title": title,
                "context": context,
                "question": question,
                "id": id_,
                "answers": {
                    "answer_start": answer_start,
                    "text": answer_text,
                },
              }
          else:
              yield id_, {
              "title": title,
              "context": context,
              "question": question,
              "id": id_,
            }
```

The `generate_dataset` is then used to create a `DataFrame` that will contain
the whole dataset framed as described above.

```python
import pandas as pd

# Create a pandas dataframe that contains all the data
df = pd.DataFrame(
    [value[1] for value in generate_dataset(ds_original)]
)
```

The result is:

```python
from IPython.display import display, HTML

def display_dataframe(df):
    display(HTML(df.to_html()))
```

```python
display_dataframe(df.head())
```

```python
df.to_csv('Q_A.csv')
```

Number of newly generated rows:

```python
n_answers = df['answers'].count()
print("Total samples:\n{}".format(n_answers))
```

### Dataset Split
The dataset has to be splitted into training set and validation set.

```python
from datasets import Dataset, DatasetDict

def split_train_validation(df, train_size):
    """
    Returns a DatasetDict with the train and validation splits.

    Parameters
    ----------
    df: Pandas.Dataframe
        Dataframe to split.
    train_size : int or float
        A number that specifies the size of the train split.
        If it is less or equal than 1, represents a percentage, else
        the train's number of samples 
    
    Returns
    -------
    DatasetDict(**dataset) : datasets.dataset_dict
        Dictionary containing as keys the train and validation split and 
        as values a dataset.

    """

    dataset = {}
    # Number of samples in df
    n_answers = df['answers'].count()
    if train_size <= 1 : s_train = n_answers * train_size 
    else: s_train= train_size
    # Count of answers by title, output is sorted asc
    df_bytitle = df.groupby(by='title')['answers'].count()
    # Cumulative sum over the DataFrame in order to select the train/validation titles
    # according to the train size
    train_title = df_bytitle[df_bytitle.sort_values().cumsum() < s_train]
    # Splitting the two dataframes
    df_train = df[df.title.isin(train_title.index.tolist())].reset_index(drop=True)
    df_validation = df[~df.title.isin(train_title.index.tolist())].reset_index(drop=True)
    # Building the two HuggingFace's datasets using train and validation dataframes
    dataset["train"]= Dataset.from_pandas(df_train)
    dataset["validation"]= Dataset.from_pandas(df_validation)

    return DatasetDict(**dataset)
```

Call `split_train_validation` in order to split in training and validation set
the previously created `DataFrame`.

```python
datasets = split_train_validation(df, 0.9)
```

The result is:

```python
datasets
```

## Preprocessing the Data

### Choosing the Model
As stated in the beginning what will be used is a transformer that has been
pretrained on a generic task. Hence, in order to finetune it, it is important to
faithfully **repeat the preprocessing steps used during the pre-training
phase**. As such it's needed to define the model that it's going to be used
straight from the preprocessing phase.<br>
Since in this context it's required to answer the questions not by generating
new text but by extracting substring from a paragraph, the ideal type of
transformer to be used is the **encoder** kind.
<figure class="image">
<img
src="https://drive.google.com/uc?export=view&id=1A9BFo4m5zuVNceYccmS_thUiUhwQfJmm">
<figcaption>Typical structure of an encoder-based transformer.</figcaption>
</figure>

From this family of transformers it has been decided to use DistilBERT.

```python
model_checkpoint = "distilbert-base-uncased"
```

### Loading the Tokenizer
The preprocessing it's handled by HuggingFace's `Tokenizer` class.<br>
This class is able to handle the preprocessing of the dataset in conformity with
the specification of each pre-trained model present in HuggingFace's model hub.
In particular they hold the vocabulary built in the pre-training phase and the
tokenization methodology used: it generally is word-based, character-based or
subword-based. DistilBERT uses the same as BERT, namely, end-to-end
tokenization: punctuation splitting and wordpiece (subword segmentation).<br>
The method `AutoTokenizer.from_pretrained` will download the appropriate
tokenizer.

```python
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

### Handling Long Sequences
The transformer models have a maximum number of tokens they are able to process
with this quantity varying depending on the architecture.<br>
A solution usually adopted in case of sequences longer than the limited amount
(other than choosing a model that can handle longer sequences) is to
**truncate** the sentence.<br>
While this approach may be effective for some tasks in this case it's **not a
valid solution** since there would be the risk of truncating out from the
context the answer to the question.<br>
In order to overcome this limitation what will be done is **sliding** the input
sentence over the model with a certain **stride** allowing a certain degree of
**overlap**. The overlap is necessary as to avoid the truncation of a sentence
in a point where an answer lies.

```python
max_length = 384 # Max length of the input sequence
stride = 128 # Overlap of the context
```

HuggingFace's tokenizer allow to perform this kind of operation by passing to
the tokenizer the argument `return_overflowing_tokens=True` and by specifying
the stride through the argument `stride`.

```python
def tokenize(tokenizer, max_length, stride, row):
    pad_on_right = tokenizer.padding_side == "right"
    
    return tokenizer(
        row["question" if pad_on_right else "context"],
        row["context" if pad_on_right else "question"],
        max_length=max_length,
        truncation="only_second" if pad_on_right else "only_first",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        stride=stride,
        padding="max_length"
    )
```

The division of a context in numerous truncated context create some issues
regarding the detection of the answer inside the context since a pair of
question-context may generate multiple pairs question-truncated context. This
implies that using `answers["answer_start"]` is not sufficient anymore. As such,
an ulterior preprocessing steps needs to be integrated in the preprocessing
pipeline: the detection of the answers in the truncated contexts.

```python
import collections

# This structure is used as an aid to the following functions since they will have to deal with a lot of start and end indexes.
Position = collections.namedtuple("Position", ["start","end"])
```

The first step is to retrieve the answer position in the original context.

```python
def get_answer_position_in_context(answers):
    # Index of the answer starting character inside the context.
    start_char = answers["answer_start"][0]
    # Index of the answer ending character inside the context.
    end_char = start_char + len(answers["text"][0])
    
    return Position(start=start_char, end=end_char)
```

Since the tokenized input sequence encodes both the question and the context it
is necessary to indentify which part of the sequence match the context.<br>
In order to complete this task the method `sequence_ids()` come into aid.<br>
In particular `sequence_ids()` tags the input tokens as `0` if they belong to
the quesiton and `1` if they belong to the context (the reverse is instead true
in the case the model pad the sequence to the left); `None` is for special
tokens.

```python
def get_context_position_in_tokenized_input(tokenized_row, i, pad_on_right):
    # List that holds for each index (up to the lenght of the tokenized input sequence)
    # 1 if its corresponding token is a context's token, 0 if it's a question's token
    # (the contrair if pad_on_right is true). Null for the special tokens.
    sequence_ids = tokenized_row.sequence_ids(i)

    # Start context's token's index inside the input sequence.
    token_start_index = sequence_ids.index(1 if pad_on_right else 0)

    # End context's token's index inside the input sequence.
    token_end_index = len(sequence_ids)-1 - list(reversed(sequence_ids)).index(1 if pad_on_right else 0)

    return Position(start=token_start_index, end=token_end_index)
```

In order to properly tag the position of an answer in a truncated context the
answer itself needs to be fully included inside the truncated context, since
partial answers may not be fully explicative, nor have grammatical consistence,
ecc...<br>
Having the start and end answer's indexes inside the original context and the
position of the truncated context inside the tokenized input sequence (which is
composed by the question and the context), what's left it to identify the
position of the answer in the tokenized and truncated context.<br>
This is done through the aid of the tokenized sequence attribute
`offset_mapping` (obtained using the argument `return_offsets_mapping=True` to
call the tokenizer) which indicates for each tokenized word its starting and
ending index in the original sequence.

```python
def get_answer_position_in_tokenized_input(offsets, char_pos, token_pos, cls_index):
    # Check if the answer fully included in the context.
    if offsets[token_pos.start][0] <= char_pos.start and offsets[token_pos.end][1] >= char_pos.end:
        # Starting token's index of the answer with respect to the input sequence.
        start_position = token_pos.start + next(i for i,v in enumerate([offset[0] for offset in offsets[token_pos.start:]]) if v > char_pos.start or i==token_pos.end+1) - 1
        # Ending token's index of the answer with respect to the input sequence.
        end_position = next(i for i,v in reversed(list(enumerate([offset[1] for offset in offsets[:token_pos.end+1]]))) if v < char_pos.end or i==token_pos.start-1) + 1

        return Position(start=start_position, end=end_position)
    else:
        return Position(start=cls_index, end=cls_index)
```

```python
def preprocess_train(tokenizer, max_length, stride):
    pad_on_right = tokenizer.padding_side == "right"

    def preprocess_train_impl(rows):
        tokenized_rows = tokenize(tokenizer, max_length, stride, rows)
        # overflow_to_sample_mapping keeps the corrispondence between a feature and the row it was generated by.
        sample_mapping = tokenized_rows.pop("overflow_to_sample_mapping")
        # offset_mapping hold for each input token it's position in the textual counterpart
        # (be it the question or the context).
        offset_mapping = tokenized_rows.pop("offset_mapping")

        tokenized_rows["start_positions"] = []
        tokenized_rows["end_positions"] = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_rows["input_ids"][i]

            # cls is a special token. It will be used to label "impossible answers".
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # One row can generate several truncated context, this is the index of the row containing this portion of context.
            sample_index = sample_mapping[i]
            answers = rows["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                pos = Position(cls_index,cls_index)
            else:
                char_pos = get_answer_position_in_context(answers)
                token_pos = get_context_position_in_tokenized_input(tokenized_rows, i, pad_on_right)
                pos = get_answer_position_in_tokenized_input(offsets, char_pos, token_pos, cls_index)

            tokenized_rows["start_positions"].append(pos.start)
            tokenized_rows["end_positions"].append(pos.end)

        return tokenized_rows
    return preprocess_train_impl
```

### Calling the Preprocessing Method
The `map` method of the DatasetDict apply a given function to each row of the
dataset (to each dataset's split).

```python
tokenized_datasets = datasets.map(preprocess_train(tokenizer, max_length, stride),
                                  batched=True,
                                  remove_columns=datasets["train"].column_names)
```

The result is:

```python
tokenized_datasets
```

## Training

As previously mentioned it's going to be used a pretrained model and then
finetuned on the task at hand. In particular DistilBERT, just like BERT, is
trained to be used mainly on masked language modeling and next sentence
prediction tasks.<br>
Since the model has already been defined during the preprocessing phase, it's
now possible to direcly download it for HuggingFace Model Hub using the
`from_pretrained` method.<br>
`AutoModel` is the class that instantiate the correct architecture based on the
model downloaded from the hub. `AutoModelForQuestionAnswering` in addition
attaches to the pretrained backbone the head needed to perform this kind of task
(which is not pretrained).

```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import zipfile

#model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

!gdown --id "1ThyHyaFwci_SXLB6jrBnm6aacN74_YCd"

with zipfile.ZipFile('squad_trained.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

model = AutoModelForQuestionAnswering.from_pretrained("squad_trained")
```

### Trainer Class Definition
The pretraining of the model will be handled by the class `Trainer`.<br>
Still, some things needs to be defined before being able to use the `Trainer`
class.<br>
The first thing is the `TrainingArguments` which specify the saving folder,
batch's size, learning rate, ecc...

```python
batch_size = 16

args = TrainingArguments(
    "squad",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01
)
```

The second and last thing to define is the data collator, which is used to batch
together sequences having different length.

```python
from transformers import default_data_collator

data_collator = default_data_collator
```

Now it's finally possible to define the Trainer class.

```python
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

### Finetuning
The method `train` of the `Trainer` class is used to trigger the finetuning
process.

```python
trainer.train()
```

Saving the model.

```python
trainer.save_model("squad-trained")
```

## Evaluation

The evaluation phase it's not straightforward and requires some additional steps
in order to perform it.<br>
In particular the output of the model are the loss and two scores indicating the
likelihood of a token being the start and end of the answer.<br>
Simply taking the argmax of both will not do since it may create unfeasible
situations: start position greater than end position and/or start position at
question (remember that the input senquence is composed by the union of the
tokenized answer and tokenized context).

### Preprocessing the Evaluation Data
Before evaluating the model some processing steps are required: all the data
necessary to avoid the aforementioned problems needs to be added to the
dataset.<br>
The problem of the answer being located inside the question is addressed by
adding the starting token of the context inside the unified input sequence.<br>
Thanks to the column `overflow_to_sample_mapping` it's also possible to have a
reference between the features and the corresponding row.

```python
def preprocess_eval(tokenizer, max_length, stride):
    pad_on_right = tokenizer.padding_side == "right"
    def preprocess_eval_impl(rows):
        # Tokenize the rows
        tokenized_rows = tokenize(tokenizer, max_length, stride, rows)

        # overflow_to_sample_mapping keeps the corrispondence between a feature and the row it was generated by.
        sample_mapping = tokenized_rows.pop("overflow_to_sample_mapping")

        # For each feature save the row that generated it.
        tokenized_rows["row_id"] = [rows["id"][sample_index] for sample_index in sample_mapping]

        # Save the start and end context's token's position inside the tokenized input sequence (composed by question plus context)
        context_pos = [get_context_position_in_tokenized_input(tokenized_rows,i,pad_on_right) for i in range(len(tokenized_rows["input_ids"]))]
        tokenized_rows["context_start"], tokenized_rows["context_end"] = [index.start for index in context_pos], [index.end for index in context_pos]

        return tokenized_rows
    return preprocess_eval_impl
```

```python
validation_features = datasets["validation"].map(
    preprocess_eval(tokenizer, max_length, stride),
    batched=True,
    remove_columns=datasets["validation"].column_names
)
```

The validation's features generated from the preprocessing are used to compute
the predictions.

```python
raw_valid_predictions = trainer.predict(validation_features)
```

Since the `Trainer` class hides the columns not used during the prediction they
have to be set back.

```python
validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
```

### Posprocessing the Evaluation Data
The aim of the posprocessing is: given the raw prediction (composed by the
likelihoods of each input token to be the starting and ending token of the
answer) the function retrieve the portion of the context's text corresponding to
the predicted answer.

`get_best_feasible_position` function select the best possible pairs of starting
and ending tokens for each answer.<br>
The problem is easily shapeable as a linear optimization problem.<br>
The function has been originally implemented by using `z3` library, but it has
been sucessively discarded because of performance issues.<br>
The used implementation can be found after `z3`'s.

```python
!pip install z3-solver
```

```python
from z3 import *

Score = collections.namedtuple("Score", ["index","score"])

def get_best_feasible_position(context_start, context_end, start_logits, end_logits):
    start_index = Int("start_index")
    end_index = Int("end_index")
    st_log = Array('st_log', IntSort(), RealSort())
    e_log = Array('e_log', IntSort(), RealSort())
    for i,sl in enumerate(start_logits):
        st_log = Store(st_log, i, sl)
    for i,el in enumerate(end_logits):
        e_log = Store(e_log, i, el)

    constraint = And(start_index < end_index,
                     start_index >= context_start,
                     end_index <= context_end)
    opt = Optimize()
    opt.add(constraint)
    opt.maximize(st_log[start_index]+e_log[end_index])
    if opt.check() == sat:
        model = opt.model()
        return Score(index=Position(start=model.evaluate(start_index).as_long(),
                                    end=model.evaluate(end_index).as_long()),
                     score=st_log[start_index]+e_log[end_index])
    else:
        raise StopIteration

```

```python
Score = collections.namedtuple("Score", ["index","score"])

def get_best_feasible_position(start_logits, end_logits, context_start, context_end, n_logits=0.15):
    #Sort logits in ascending order
    sorted_start_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)[:int(len(start_logits)*n_logits)]
    sorted_end_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)[:int(len(end_logits)*n_logits)]

    # Associate the positions of each pair of start and end tokens to their score and sort them in descending order of score
    sorted_scores = collections.OrderedDict(
                            sorted({Position(start=i, end=j):sl+el for i,sl in sorted_start_logit for j,el in sorted_end_logit}.items(),
                                    key=lambda x: x[1],
                                    reverse=True)
                    )
    
    # Return the position of the pair of higher score that respects the consistency constraints
    return next(Score(index=pos, score=score) for pos,score in sorted_scores.items() \
                if pos.start <= pos.end and pos.start >= context_start and pos.end <= context_end)
```

`map_feature_to_row` uses the `row_id` that has been added during the
preprocessing step in order to create a corrispondence between a feature and the
row it belong to.

```python
def map_feature_to_row(dataset, features):
    # Associate rows' id with an index
    row_id_to_index = {k: i for i, k in enumerate(dataset["id"])}
    features_per_row = collections.defaultdict(list)
    # Create a corrispondence beween the previously computed rows' index with
    # the index of the features that belong to the said rows
    for i, feature in enumerate(features):
        features_per_row[row_id_to_index[feature["row_id"]]].append(i)

    return features_per_row
```

The `postprocess_eval` function use the two function defined above and for each
raw prediction returns a portion of context's text that best match it taking
into account:
- The logits values outputted by the model.
- The consistency constraints mentioned above.

```python
from tqdm.notebook import tqdm

def postprocess_eval(dataset, features, raw_predictions, verbose=True):
    all_start_logits, all_end_logits = raw_predictions

    # Map the dataset's rows to their corresponding features.
    features_per_row = map_feature_to_row(dataset, features)

    predictions = collections.OrderedDict()

    if verbose:
        print(f"Post-processing {len(dataset)} dataset predictions split into {len(features)} features.")

    for row_index, row in enumerate(tqdm(dataset)):
        valid_answers = []

        # Indices of the features associated to the current row.
        feature_indices = features_per_row[row_index]
        
        context = row["context"]
        # Loop on the features associated to the current row.
        for feature_index in feature_indices:
            context_start = features[feature_index]["context_start"]
            context_end = features[feature_index]["context_end"]

            offsets = features[feature_index]["offset_mapping"]

            # Computation of the answer from the raw preditions.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            try:
                valid_answers.append(get_best_feasible_position(start_logits, end_logits, context_start, context_end))
            except StopIteration:
                continue

        # For each row use as answer the best candidate generated by the row's features
        if len(valid_answers) > 0:
            answer_pos = sorted(valid_answers, key=lambda x: x.score, reverse=True)[0].index
            answer = context[offsets[answer_pos.start][0]: offsets[answer_pos.end][1]]
        # In case no candidates are found return an empty string
        else:
            print("Not found any consistent answer's start and/or end")
            answer = ""

        predictions[row["id"]] = answer

    return predictions
```

Calling the post-processing function over the validation set.

```python
validation_predictions = postprocess_eval(datasets["validation"],
                                          validation_features,
                                          raw_valid_predictions.predictions)
```

### Compute Metrics
The metrics that are those provided from HuggingFace for the squad dataset:
exact match and f1 score.

```python
from datasets import load_metric

metric = load_metric("squad")
```

```python
formatted_predictions = [{"id": k, "prediction_text": v} for k, v in validation_predictions.items()]
references = [{"id": r["id"], "answers": r["answers"]} for r in datasets["validation"]]

metric.compute(predictions=formatted_predictions, references=references)
```

###  Error analysis
In order to analyze what kind of errors the model made, the mistaken predictions
should first be retrieved.<br>
With "mistaken predictions" are intended those predictions that do not exactly
match with the ground truth.

```python
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
```

```python
actual_match = pd.DataFrame([{"question":row["question"], "context":row["context"], "ground_truth":row["answers"]["text"][0], "prediction":validation_predictions[row["id"]]}
                       for row in datasets["validation"] \
                       if normalize_answer(row["answers"]["text"][0]) == normalize_answer(validation_predictions[row["id"]])])
```

```python
display_dataframe(actual_match.head(30))
```

```python
errors = pd.DataFrame([{"question":row["question"], "context":row["context"], "ground_truth":row["answers"]["text"][0], "prediction":validation_predictions[row["id"]]}
                       for row in datasets["validation"] \
                       if normalize_answer(row["answers"]["text"][0]) != normalize_answer(validation_predictions[row["id"]])])
```

Total number of mistaken predictions.

```python
print("Wrong answers: {}/{}".format(len(errors),len(datasets["validation"])))
```

In order to check what kind of mistakes the model made, some of the errors will
be displayed.<br>
First 30 errors:

```python
# display_dataframe is defined in the Datast Creation paragraph
display_dataframe(errors.head(30))
```

Random 30 errors:

```python
display_dataframe(errors.sample(frac=1).reset_index(drop=True).head(30))
```

Retrieve an error by querying by question.

```python
def get_error(errors, question):
    return errors[errors['question']==question]
```

```python
display_dataframe(get_error(errors, 'What genre of movie did Beyonce star in with Cuba Gooding, Jr?'))
```


