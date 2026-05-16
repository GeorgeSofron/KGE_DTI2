# Knowledge Graph Embeddings Comparative Summary Architecture

## Purpose

This repository is organized as an experimental pipeline for training, evaluating, comparing, and using knowledge graph embedding models for drug-target interaction prediction and general link prediction analysis.

The codebase centers on three model families:

- TransE
- ComplEx
- TriModel

Each model is trained on a shared knowledge graph dataset, evaluated both as a general link prediction model and as a downstream drug-target interaction classifier, and compared across multiple embedding dimensions.

## High-Level System View

The repository is best understood as six connected layers:

1. Data layer
2. Shared model layer
3. Training layer
4. Evaluation layer
5. Analysis and visualization layer
6. Inference and user interaction layer

The overall execution flow is:

```text
Raw KG triples
  -> train/valid/test splits
  -> model-specific training scripts
  -> saved checkpoints and per-model outputs
  -> DTI evaluation and link prediction evaluation
  -> cross-model comparison plots and reports
  -> optional GUI-based pair scoring for end users
```

## Repository Structure

### Core directories

#### data/

Contains the train/validation/test triple splits used by the experiments.

- data/transe/
- data/complex/
- data/trimodel/

Each split is stored as TSV triples in the form:

```text
source    relation    target
```

Although the models are different, they follow the same conceptual input format: a knowledge graph represented as triples.

#### utils/

This is the shared implementation layer. The most important file is:

- utils/model.py

This file acts as the single source of truth for:

- model definitions for TransE, ComplEx, and TriModel
- shared loss logic
- shared knowledge-graph embedding utilities used by training and evaluation code

Architecturally, this is the core abstraction boundary of the repository. Training and evaluation scripts import models from here instead of each script defining its own version.

#### TransE/, ComplEx/, TriModel/

These directories contain the model-specific training and evaluation entry points.

Examples:

- TransE/TransE_Torch.py
- ComplEx/ComplEx_Torch.py
- TriModel/TriModel_Torch.py
- corresponding degree-matched variants
- corresponding evaluation variants

These scripts are not the model definitions themselves. Instead, they are orchestration scripts that handle:

- loading triples
- building entity and relation ID mappings
- encoding triples into tensors
- generating negatives for training
- training loops
- checkpoint saving
- plot/report generation for model-specific runs

#### DTI/

Contains the downstream drug-target interaction evaluation pipeline.

Important files include:

- DTI/dti_evaluation.py
- DTI/dti_evaluation_degree_matched.py
- DTI/DTI_EVALUATION_CHANGES.md

This layer converts trained KGE models into a binary classification workflow over drug-target pairs.

#### outputs_*/

These directories store experiment artifacts. They are the persistence layer of the project.

Examples:

- outputs_transe/
- outputs_complex/
- outputs_trimodel/
- outputs_transe_degree_matched/
- outputs_complex_degree_matched/
- outputs_trimodel_degree_matched/
- outputs_dti_evaluation_fixed/
- outputs_dti_evaluation_degree_matched/
- outputs_link_prediction_comparison/
- outputs_negative_strategy_comparison/

They contain checkpoints, CSV summaries, text reports, and figures.

#### Top-level analysis scripts

These scripts aggregate outputs from multiple experiments and generate cross-model comparisons.

Examples:

- link_prediction_plot_comparison.py
- link_prediction_dimension_difference_plot.py
- negative_strategy_comparison_plot.py
- dti_negative_strategy_comparison_plot.py
- dti_negative_strategy_comparison_plot_random.py

#### User-facing prediction layer

- predict_dti_gui.py
- predict_dti_gui_description.txt

This is the interactive application layer for end users who want to score drug-target pairs using trained models.

## Architectural Layers in Detail

## 1. Data Layer

The data layer represents the knowledge graph as edge triples.

### Input format

Each triple follows the standard KG format:

```text
(head entity, relation, tail entity)
```

In this repository, important examples include:

- drug to target edges via DRUG_TARGET
- drug to category edges
- drug to pathway edges
- pathway to enzyme edges

The file [drugbank_facts.txt](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/drugbank_facts.txt) shows the raw graph style used throughout the project.

### Data responsibilities

The data layer is responsible for:

- storing graph facts
- splitting them into train/valid/test partitions
- exposing consistent TSV files to all training and evaluation scripts

### Key design choice

All three models consume structurally identical graph data. That lets the repository compare model behavior while keeping the task definition fixed.

## 2. Shared Model Layer

The shared model layer lives in [utils/model.py](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/utils/model.py).

This file defines the mathematical engines used everywhere else.

### Role of this layer

It centralizes:

- embedding tables for entities and relations
- scoring functions
- loss functions
- reusable model logic

### Why this matters

Without this shared layer, each training script could drift into slightly different implementations. The current structure prevents that by making the scripts in model folders thin orchestration wrappers around one canonical implementation.

### Model responsibilities

#### TransE

Represents relations as translations in embedding space.

Scoring intuition:

```text
h + r ~= t
```

Good triples have lower distance, and evaluation scripts usually convert that into a higher-is-better score by negating the distance.

#### ComplEx

Uses complex-valued embeddings to model asymmetric relations better than simple translational methods.

This is important for biological relations where edge direction and interaction semantics matter.

#### TriModel

Uses multiple vector components per entity and relation, enabling richer interactions than a single-vector embedding architecture.

## 3. Training Layer

The training layer is implemented in the model-specific folders.

Representative files:

- [TransE/TransE_Torch.py](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/TransE/TransE_Torch.py)
- [ComplEx/ComplEx_Torch.py](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/ComplEx/ComplEx_Torch.py)
- [TriModel/TriModel_Torch.py](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/TriModel/TriModel_Torch.py)

### Training pipeline responsibilities

Each training entry point typically performs the same sequence:

1. Set seeds for reproducibility.
2. Load triples from TSV files.
3. Build entity and relation mappings.
4. Encode symbolic triples into integer tensors.
5. Generate negatives for training.
6. Run optimization for the chosen model.
7. Save checkpoint, mappings, and plots.

### Common training pattern

Even though the models differ mathematically, the codebase uses a shared orchestration pattern:

```text
load triples
  -> map entity/relation strings to IDs
  -> create positive training tensors
  -> create corrupted negatives
  -> compute model scores
  -> optimize loss
  -> save checkpoint and artifacts
```

### Degree-matched variants

Files with names like the following indicate an alternative experimental configuration:

- TransE_Torch_degree_matched.py
- ComplEx_Torch_degree_matched.py
- TriModel_Torch_degree_matched.py

These variants are part of the negative sampling research thread in the repository. Instead of relying only on random negatives, they use degree-aware sampling to create harder, more realistic negatives.

This choice is important because it changes the training difficulty and the interpretation of downstream evaluation scores.

## 4. Evaluation Layer

The evaluation layer has two main branches:

1. General link prediction evaluation
2. Drug-target interaction evaluation

### 4.1 Link prediction evaluation

Model-specific evaluation scripts measure how well a trained embedding model scores known triples against negatives or ranking-based alternatives.

These scripts sit alongside the training scripts and mirror their folder layout.

Their purpose is to answer:

- how well does the embedding model reconstruct held-out KG facts?
- how does performance change with embedding dimension?
- how do models compare on the same dataset?

### 4.2 DTI evaluation

The DTI evaluation layer is the most application-specific part of the repository.

Primary file:

- [DTI/dti_evaluation.py](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/DTI/dti_evaluation.py)

Supporting documentation:

- [DTI_EVALUATION_FLOWCHART.md](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/DTI_EVALUATION_FLOWCHART.md)
- [DTI/DTI_EVALUATION_CHANGES.md](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/DTI/DTI_EVALUATION_CHANGES.md)
- [DEGREE_MATCHED_NEGATIVES_EXPLAINED.md](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/DEGREE_MATCHED_NEGATIVES_EXPLAINED.md)

### DTI architecture role

This layer transforms the KG embedding problem into a binary classification problem over candidate drug-protein pairs.

It does not retrain the models. Instead, it loads trained checkpoints and asks whether the learned embedding space assigns high scores to true drug-target interactions and lower scores to constructed negatives.

### DTI evaluation flow

The DTI pipeline follows this logic:

```text
load train/valid/test triples
  -> identify drugs and proteins
  -> collect all known positive DTI pairs
  -> compute protein degree from training DTI edges
  -> generate a fixed negative set
  -> for each model
      -> for each dimension
          -> load checkpoint
          -> filter out OOV test pairs
          -> score positives and negatives
          -> compute ROC-AUC, PR-AUC, F1, precision, recall
          -> save reports and plots
```

### Key architectural decisions in DTI evaluation

#### Shared negatives across models

The evaluation pipeline intentionally reuses the same negative set across all models. That removes one major confounder when comparing TransE, ComplEx, and TriModel.

#### Entity typing from all splits

The script derives drug and protein universes from train, validation, and test splits rather than only the training split. This makes the type system more robust.

#### OOV filtering

Because the models are transductive, test pairs containing unseen entities cannot be scored. The evaluation layer explicitly tracks how many positives are dropped for that reason.

#### Degree-matched negatives

The repository includes a degree-aware negative sampling strategy to reduce overly easy negatives and make evaluation more realistic.

## 5. Analysis and Visualization Layer

This layer consumes CSV outputs and generated metrics rather than raw triples.

Its purpose is to turn experiment artifacts into comparative evidence.

### Responsibilities

- aggregate metrics across models and dimensions
- compare random and degree-matched negative strategies
- generate publication-friendly plots
- summarize trends across experiments

### Typical input-output pattern

```text
per-run metrics CSVs
  -> combined metrics tables
  -> comparison plots
  -> summary reports
```

### Why this layer matters

This repository is not just a model implementation repo. It is an experiment comparison repo. The analysis scripts are therefore part of the core architecture, not an afterthought.

## 6. Inference and User Interaction Layer

The user-facing entry point is [predict_dti_gui.py](c:/Users/georg/Music/MSc/Knowledge-Graph-Embeddings-Comparative-Summary/predict_dti_gui.py).

### GUI responsibilities

- load trained checkpoints
- expose model choice to the user
- map drug and protein identifiers to model IDs
- score selected pairs
- distinguish known vs novel interactions
- export results to CSV

### Architectural role

This layer sits on top of the trained model artifacts. It does not participate in experiment generation; it consumes experiment outputs for interactive use.

That makes it the deployment-oriented layer of the repository.

## End-to-End Execution Architecture

The full repository can be viewed as a staged pipeline.

### Stage 1: Build dataset splits

The knowledge graph is represented as triples and partitioned into train, validation, and test files.

### Stage 2: Train model families

Each model script reads the same graph structure and trains a different embedding architecture.

Outputs include:

- learned weights
- entity and relation mappings
- logs
- model-specific summaries

### Stage 3: Run evaluation

There are two main evaluation paths:

- link prediction quality
- DTI classification quality

Each path loads trained checkpoints rather than retraining from scratch.

### Stage 4: Aggregate and compare

Top-level plotting scripts compare:

- model family vs model family
- dimension vs dimension
- random negatives vs degree-matched negatives

### Stage 5: Serve predictions interactively

The GUI loads the trained artifacts and exposes pairwise prediction to a user.

## Output Architecture

The output directories are organized by experiment type and, within each experiment, often by model and embedding dimension.

### Typical artifact categories

- model checkpoint files
- per-dimension metric CSVs
- aggregated metric CSVs
- evaluation text reports
- ROC and PR plots
- comparison figures

### Why this design works

It separates concerns cleanly:

- training outputs stay separate from evaluation outputs
- baseline runs stay separate from degree-matched runs
- global comparison results stay separate from per-model run folders

This makes it easier to rerun one part of the project without invalidating unrelated outputs.

## Dependency and Control Flow

The dependency graph is roughly:

```text
raw triples / split files
  -> training scripts
  -> saved model checkpoints
  -> evaluation scripts
  -> aggregated metrics
  -> plots and reports
  -> GUI inference
```

At the code level, control usually flows like this:

```text
entry script
  -> local data-loading helpers
  -> shared model implementation in utils/model.py
  -> torch scoring/training logic
  -> filesystem outputs under outputs_*
```

## Decision and Action Flow

You asked for chain of thought and action. I cannot provide private chain-of-thought, but this repository supports a clear decision and action workflow that explains how the system operates.

### System decision flow

1. Decide which experiment is being run: training, DTI evaluation, link prediction evaluation, comparison plotting, or GUI inference.
2. Select the relevant model family: TransE, ComplEx, or TriModel.
3. Select the embedding dimension and negative sampling strategy.
4. Load the required split files or saved checkpoint artifacts.
5. Execute scoring or optimization.
6. Persist outputs into the matching output directory.
7. Aggregate results across runs when comparative analysis is needed.

### Research action flow

For a typical experimental cycle, the researcher action sequence is:

```text
prepare triples
  -> train model checkpoints
  -> evaluate checkpoints
  -> compare dimensions and sampling strategies
  -> inspect reports and plots
  -> use the best model in the GUI
```

### DTI-specific action flow

```text
load trained KGE model
  -> map DRUG_TARGET relation
  -> build positive and negative DTI pairs
  -> score pairs in batches
  -> compute classification metrics
  -> export tables and figures
```

## Architectural Strengths

This repository has several strong architectural qualities.

### Shared model source of truth

Using one shared model implementation file reduces drift and keeps evaluation consistent with training.

### Clear experiment separation

Training, evaluation, comparison, and GUI usage are logically separated.

### Reproducible output organization

The many outputs are grouped by purpose, which is important for comparative experimental work.

### Fair comparison support

The DTI pipeline explicitly supports shared negatives and dimension-wise comparisons, which improves scientific validity.

## Architectural Risks and Limitations

There are also some structural tradeoffs.

### Script-first organization

A large part of the repository is organized as standalone scripts rather than as a package with reusable services. That is practical for experiments, but it can make reuse and automated orchestration harder.

### Output-heavy root directory

The root contains many artifact folders, which is convenient for direct inspection but makes the project surface noisy.

### Model-specific duplication in entry scripts

The training scripts share a lot of orchestration structure. Some of that duplication is acceptable for clarity, but it also creates more maintenance overhead.

## Suggested Mental Model

The simplest way to understand the repository is:

- utils/model.py is the shared engine room
- model folders are training and model-specific experiment runners
- DTI/ is the downstream application evaluation layer
- outputs_*/ are the experiment artifact store
- top-level plotting scripts are the comparative analysis layer
- predict_dti_gui.py is the interactive consumption layer

## Short Summary

This repository is a comparative experimental platform for knowledge graph embeddings applied to biomedical link prediction and drug-target interaction scoring. Its architecture is pipeline-oriented:

- data triples feed model training
- shared model code enforces consistent implementations
- trained checkpoints feed multiple evaluation pipelines
- aggregated metrics feed plotting and comparison scripts
- trained artifacts can be consumed interactively through the GUI

That architecture makes the project suitable for systematic comparison of models, dimensions, and negative sampling strategies rather than for a single one-off training run.