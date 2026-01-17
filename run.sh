#!/bin/bash
set -e
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python main.py "data=atmospheric_physics" > logs/atmospheric_physics.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main.py "data=new_york_taxi"        > logs/new_york_taxi.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python main.py "data=simulation_sine"      > logs/simulation_sine.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python main.py "data=traffic_FromNewstoForecast" > logs/traffic_FromNewstoForecast.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python main.py "data=electricity_accomodation" > logs/electricity_accomodation.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python main.py "data=weather_TimeCAP"          > logs/weather_TimeCAP.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python main.py "data=weather_TimeCAPnew"   > logs/weather_TimeCAPnew.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python main.py "data=weather_TimeCAPsan"   > logs/weather_TimeCAPsan.log 2>&1 &
wait
echo "All jobs launched and finished."
