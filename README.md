# poverty-targeting

Predicting household multidimensional deprivation (global MPI, Ghana DHS 2022) from a short proxy-means-test questionnaire, served as an API and a targeting dashboard.

## Live API

A public demonstration deployment runs on Azure Container Apps (Spain Central):

- Interactive docs: https://poverty-targeting-api.mangograss-dd837579.spaincentral.azurecontainerapps.io/docs
- Health check: https://poverty-targeting-api.mangograss-dd837579.spaincentral.azurecontainerapps.io/health

It scales to zero when idle, so the first request after a quiet period can take up to a minute.
This is a demonstration service: please don't send real households' personal information.