# Racing for the Future
Repository for ["Racing for the Future: Wealth Competition and Interest Rates in Anticipation of TAI"](https://calebmaresca.github.io/docs/papers/AI_Interest_Rates.pdf)

The rapid progress of development in the field of artificial intelligence may profoundly reshape the global economy by both increasing productivity and automating away many jobs. This paper explores how households adjust their economic behavior today in anticipation of transformative AI (TAI). Building on previous research, I introduce a novel mechanism where the future reallocation of labor from humans to AI systems owned by wealthy households creates a zero-sum contest for control over AI resources—driving changes in current savings decisions and asset prices.

## Overview

This repository contains the code used to generate the results in the paper. The model simulates how expectations of Transformational AI affect current economic behavior through a modified neoclassical growth framework.

## Installation

```bash
git clone https://github.com/yourusername/tai-capital-race.git
cd tai-capital-race
pip install -r requirements.txt
```

To download the Metaculus AI timelines data, navigate to https://www.metaculus.com/questions/5121/date-of-artificial-general-intelligence/ and click `Download Question data`. Create a `data` folder in the root of this repo and move the .csv files there.

## Usage

To generate Figure 1 in the paper and save the calibrated probabilities, run `fit_tai_dist.py`.

To run the main analysis and produce Table 1 and the remaining figures, run `analysis.py`.
