---
title: Poverty Targeting
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Estimate MPI poverty likelihood and explore targeting budgets
---

# Poverty targeting dashboard

Estimate how likely a household in Ghana is to be **multidimensionally poor** (global MPI)
from a short questionnaire, see the main reasons, and explore how a programme's budget
changes who it reaches.

The model runs in a separate prediction API; this dashboard calls it over HTTPS.
Code, methods, validation and model card: https://github.com/Mbibachris/poverty-targeting

This is a demonstration: please don't enter real households' personal information.
Estimates support, and never replace, human judgement.