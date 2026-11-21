#!/usr/bin/env bash
# Women's E-Commerce Clothing Reviews (Nick Brooks "nicapotato", 2018; CC0, originally on Kaggle). GitHub mirror of the same 23,486 rows.
set -e
mkdir -p data
[ -f data/reviews.csv ] || curl -fsSL -o data/reviews.csv \
  "https://raw.githubusercontent.com/AFAgarap/ecommerce-reviews-analysis/master/Womens%20Clothing%20E-Commerce%20Reviews.csv"
echo "data/reviews.csv ready"
