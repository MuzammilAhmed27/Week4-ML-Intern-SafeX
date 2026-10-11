# Synthetic Museum Dataset — Data Sheet

## Source
Generated locally by scripts/train.py. No real museum collections, visitors, or private records were used.

## Purpose
Demonstrate a machine-learning inference service with two model versions.

## Features
- age_years: estimated age in years
- size_cm: size in centimetres
- estimated_value_usd: synthetic estimated value in USD
- organic: binary indicator
- has_inscription: binary indicator
- excavation_find: binary indicator
- natural_specimen: binary indicator
- artwork: binary indicator
- category: synthetic target label

## Label-generation rules
Natural specimens map to Natural History; artwork maps to Art; excavation finds or inscribed objects map to Archaeology; other records map to History.

## Limitations and risks
The dataset is synthetic and labels follow predefined rules. Model results do not demonstrate real-world performance. Values must not be used for financial or insurance decisions.

## Licence
Original synthetic data generated for this demonstration; no third-party dataset was used. Add a project-level licence if you publish the repository.