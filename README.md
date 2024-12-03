# ACES_emergent_scheduler

The simulation allows you to test different scheduling methods (Random, Best, and Bottom-Up) for assigning pods to workers and evaluate the cost and quality of the allocation in terms of utilization and satisfaction rates.

## Requirements

| **Package**  | **Version** |
|--------------|-------------|
| Python       | 3.11.7      |
| NumPy        | 1.26.4      |
| Mesa         | 2.2.4       |

## Configuration

The values for **`alpha`**, **`beta`**, and **`gamma`** can be configured in the `config.yaml` file.

## Running the Simulation

Execute the script `scheduler-demo2.py`. The program will simulate the allocation process over multiple traffic intensities **`λ`**. In the script **`thresholds`** take in the values **`alpha`** and **`beta`** from the config file and **`Gamma`** is taken from the parameter **`gamma`**.

The executed algorithms **`random`**, **`best`**, and **`bottum_up`** can be accessed in the file `algorithms.py`.

## Obtaining Results

- Utilization and Satisfaction Rate:
  - The script calculates CPU/RAM utilization and satisfaction rates for rigid and elastic pods
  - Generated graphs can be found in the working directory
- Cost/Quality Metrics
  - The script outputs the utilization rates and satisfaction percentages for each model.
