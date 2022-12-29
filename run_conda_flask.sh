#!/bin/bash
cd /root/codebase/model_api
source /root/anaconda3/etc/profile.d/conda.sh
conda activate movie_recommender_flask
python -m flask run --host=0.0.0.0