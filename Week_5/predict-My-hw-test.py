#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9690/predict_hw'

client = {"job": "retired", "duration": 445, "poutcome": "success"}

response = requests.post(url, json=client).json()
print(response)
