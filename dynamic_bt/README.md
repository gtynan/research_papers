# dynamic_bt

## Overview

Python implementation of the NBA section of [this paper](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9876.2012.01046.x).

## Structure

Constants declared in paper such as **home smoothing param** are located in `conf/base/parameters.yml`. 

Data is saved as pipeline is run, initial dataset is saved in `03_primary` (can also be rescraped by uncommenting `scrape_season_data` in `src/dynamic_bt/pipelines/dynamic_bt`).

Locations data will be saved is mapped out in `conf/base/catalog.yml`

## Running

```shell
kedro run
```
