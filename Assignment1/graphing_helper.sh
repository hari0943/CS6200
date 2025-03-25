#!/bin/bash

n_values=(10 100 200 300 500 1000 2000 3000)
k=10
output_file="simulation_results.csv"

echo "Experiments,Serial Time (s),Serial CUDA Time (s),Serial Pi Estimate,Serial Pi Estimation Error (%),Parallel Time (s),Parallel CUDA Time (s),Parallel Pi Estimate,Parallel Pi Estimation Error (%),Average Speedup" > "$output_file"

for n in "${n_values[@]}"; do
    total_serial_time=0
    total_serial_cuda_time=0
    total_serial_pi_estimate=0
    total_serial_percentage_delta=0
    total_parallel_time=0
    total_parallel_cuda_time=0
    total_parallel_pi_estimate=0
    total_parallel_percentage_delta=0
    for ((i=1; i<=k; i++)); do
        output=$(./simulation "$n")
        serial_output=$(echo "$output" | head -n 1)
        parallel_output=$(echo "$output" | tail -n 1)
    
        serial_time=$(echo "$serial_output" | awk '{print $1}')
        serial_cuda_time=$(echo "$serial_output" | awk '{print $2}')
        serial_pi_estimate=$(echo "$serial_output" | awk '{print $3}')
        serial_percentage_delta=$(echo "$serial_output" | awk '{print $4}')
    
        parallel_time=$(echo "$parallel_output" | awk '{print $1}')
        parallel_cuda_time=$(echo "$parallel_output" | awk '{print $2}')
        parallel_pi_estimate=$(echo "$parallel_output" | awk '{print $3}')
        parallel_percentage_delta=$(echo "$parallel_output" | awk '{print $4}')

        total_serial_time=$(echo "$total_serial_time + $serial_time" | bc)
        total_serial_cuda_time=$(echo "$total_serial_cuda_time + $serial_cuda_time" | bc)
        total_serial_pi_estimate=$(echo "$total_serial_pi_estimate + $serial_pi_estimate" | bc)
        total_serial_percentage_delta=$(echo "$total_serial_percentage_delta + $serial_percentage_delta" | bc)

        total_parallel_time=$(echo "$total_parallel_time + $parallel_time" | bc)
        total_parallel_cuda_time=$(echo "$total_parallel_cuda_time + $parallel_cuda_time" | bc)
        total_parallel_pi_estimate=$(echo "$total_parallel_pi_estimate + $parallel_pi_estimate" | bc)
        total_parallel_percentage_delta=$(echo "$total_parallel_percentage_delta + $parallel_percentage_delta" | bc)
    done
    avg_serial_time=$(echo "$total_serial_time / $k" | bc -l)
    avg_serial_cuda_time=$(echo "$total_serial_cuda_time / $k" | bc -l)
    avg_serial_pi_estimate=$(echo "$total_serial_pi_estimate / $k" | bc -l)
    avg_serial_percentage_delta=$(echo "$total_serial_percentage_delta / $k" | bc -l)
    avg_parallel_time=$(echo "$total_parallel_time / $k" | bc -l)
    avg_parallel_cuda_time=$(echo "$total_parallel_cuda_time / $k" | bc -l)
    avg_parallel_pi_estimate=$(echo "$total_parallel_pi_estimate / $k" | bc -l)
    avg_parallel_percentage_delta=$(echo "$total_parallel_percentage_delta / $k" | bc -l)
    avg_speedup=$(echo "$avg_serial_time / $avg_parallel_time" | bc -l)
    echo "$n,$avg_serial_time,$avg_serial_cuda_time,$avg_serial_pi_estimate,$avg_serial_percentage_delta,$avg_parallel_time,$avg_parallel_cuda_time,$avg_parallel_pi_estimate,$avg_parallel_percentage_delta,$avg_speedup" >> "$output_file"
done

echo "Simulation results saved to $output_file."
