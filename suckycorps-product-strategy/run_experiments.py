from batch_simulator import run_agents_in_environment
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

#######################################
# Two options about what to do with the agent's log messages in absence 
# of a GUI pane
def log_to_console(msg):
    print(msg)

def log_null(msg):
    pass

########################################

# Write all batch results to this file
default_output_file_name = 'simulation_results.csv'

## All possible agents and their recon configurations

agents = [ ("agents/reactiveagent.py",    "NoSenseAgent"),
           ("agents/reactiveagent.py",    "SensingAgent"),
           ("agents/worldmodelagent.py",  "WorldModelAgent"),
           ("agents/planningagent.py",    "OmniscientAgent"),
         ]

recon = {"NoSenseAgent": 'None',
         "SensingAgent": 'None',
         "WorldModelAgent": 'None',
         "OmniscientAgent": 'Full'}

#############################################################################
# Show a single example (agent + dirt density + wall density + num samples)
#  of running an experiment.
#
# EXAMPLE ONLY!  This just shows you how to set parameters in order to run
# your experiments!
'''
def run_agents_example():
    dirt_density = 0.1
    wall_density = 0.3
    num_samples = 10
    write_results_to_console = True
    demo_agents = [agents[1], agents[2]]   # Sensing agent and world model agent
    battery_capacity = 1000               # This is twice the default battery 
                                            # capacity -- most of your experiments
                                            # should use the default, you get the 
                                            # default by setting battery_capacity to None
    run_agents_in_environment(dirt_density, 
                              wall_density, 
                              demo_agents, 
                              recon,
                              battery_capacity,
                              log_to_console,   # Print agent log messages to console
                              num_samples, 
                              default_output_file_name, 
                              write_results_to_console)

#  Run the demo!
run_agents_example()
'''

#############################################################################
# Part  1 of Analysis
#   NoSenseAgent, SensingAgent, WorldModelAgent, OmniscientAgent
def run_agents_full_experiment():
    # Testing different (Low, Medium, High) Densities for Dirt and Wall
    dirt_densities = [0.1, 0.5, 0.9]
    wall_densities = [0.1, 0.5, 0.9]
    num_samples = 10
    write_results_to_console = True

    for dirt_density in dirt_densities:
        for wall_density in wall_densities:
            for agent in agents:
                run_agents_in_environment(
                    dirt_density,
                    wall_density,
                    [agent],
                    recon,
                    None,  # Default Battery Capacity
                    log_to_console,
                    num_samples,
                    default_output_file_name,
                    write_results_to_console
                )

# Run the full experiment instead of the demo
run_agents_full_experiment()

##################################################################################
# Part 1: Calculating Average Score and Variance
# Calculate Average Score and Variance
print("\nCalculating performance statistics...\n")

# Read the simulation results from the output file
df = pd.read_csv(default_output_file_name, header=None, names=["Agent", "Dirt Density", "Wall Density", "Score"])

# Convert Score to numeric in case it's read as string
df["Score"] = pd.to_numeric(df["Score"])

# Compute the mean and variance for each unique (Agent, Dirt Density, Wall Density) combination
summary_df = df.groupby(["Agent", "Dirt Density", "Wall Density"]).agg(
    Avg_Score=("Score", "mean"),
    Score_Variance=("Score", "var")
).reset_index()

# Print summary to console
print(summary_df)

#############################################################################
# Part 2 Of Analysis

default_output_file_name_part2 = 'simulation_results_part2.csv'

# Experiment 2: Varying Battery Capacity for WorldModelAgent
def run_battery_capacity_experiment():
    dirt_densities = [0.1, 0.5, 0.9]
    wall_densities = [0.1, 0.5, 0.9]
    battery_capacities = [500, 1000, 2000]  # Different battery capacities
    num_samples = 10
    write_results_to_console = True
    agent = [agents[2]]  # WorldModelAgent

    # Check if the file exists, if it does not, create it and write the headers
    import os
    file_exists = os.path.isfile(default_output_file_name_part2)

    for battery_capacity in battery_capacities:
        for dirt_density in dirt_densities:
            for wall_density in wall_densities:
                run_agents_in_environment(
                    dirt_density,
                    wall_density,
                    agent,
                    recon,
                    battery_capacity,
                    log_to_console,
                    num_samples,
                    default_output_file_name_part2,
                    write_results_to_console
                )

# Run the battery capacity experiment
run_battery_capacity_experiment()

# Read the simulation results from the output file with debugging
try:
    df = pd.read_csv(default_output_file_name_part2, header=None)
    print("\nRaw Data from CSV:\n", df.head())  # Debugging: Print first few rows
except FileNotFoundError:
    print(f"Error: {default_output_file_name_part2} not found.")
    exit()

# Assign correct column names
df.columns = ["Battery Capacity", "Dirt Density", "Wall Density", "Score"]

# Ensure numeric conversion
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df["Battery Capacity"] = pd.to_numeric(df["Battery Capacity"], errors="coerce")

df = df.dropna(subset=["Battery Capacity", "Score"])
df["Battery Capacity"] = df["Battery Capacity"].astype(int)


# Debugging: Check if NaN values exist
print("\nChecking for NaN values:\n", df.isna().sum())

# Remove any rows with NaN values
df = df.dropna()

# Compute the mean and variance for each unique combination
summary_df = df.groupby(["Battery Capacity", "Dirt Density", "Wall Density"]).agg(
    Avg_Score=("Score", "mean"),
    Score_Variance=("Score", "var")
).reset_index()

# Debugging: Check if summary_df is empty
if summary_df.empty:
    print("\nError: Summary DataFrame is empty. Check if data was recorded properly in CSV.")
else:
    # Print summary to console
    print("\nPerformance Statistics Summary:\n", summary_df)

    # Display table for analysis
    print("\nDisplaying Performance Statistics Table for different battery levels in WorldModelAgent:\n")
    print(summary_df.to_string(index=False))

##################################################################################################
# Calculation

# Calculate Average Score and Variance for different
print("\nCalculating performance statistics for different battery levels for WorldAgentModel...\n")

# Read the simulation results from the output file
df = pd.read_csv(default_output_file_name_part2, header=None,
                 names=["Battery Capacity", "Dirt Density", "Wall Density", "Score"])

# Convert Score to numeric in case it's read as string
df["Score"] = pd.to_numeric(df["Score"])

# Compute the mean and variance for each unique (Agent, Dirt Density, Wall Density) combination
summary_df = df.groupby(["Battery Capacity", "Dirt Density", "Wall Density"]).agg(
    Avg_Score=("Score", "mean"),
    Score_Variance=("Score", "var")
).reset_index()

# Print summary to console
print(summary_df)

# Display table for visual analysis
print("\nDisplaying Performance Statistics Table for different battery levels in WorldModelAgent:\n")
print(summary_df.to_string(index=False))

##################################################################################################
# Visualization of Analysis

default_output_file_name = 'simulation_results.csv'
default_output_file_name_part2 = 'simulation_results_part2.csv'

# Read Experiment 1 Results
df1 = pd.read_csv(default_output_file_name, header=None, names=["Agent", "Dirt Density", "Wall Density", "Score"])
df1["Score"] = pd.to_numeric(df1["Score"])

# Read Experiment 2 Results (Battery Impact)
df2 = pd.read_csv(default_output_file_name_part2, header=None, names=["Battery Capacity", "Dirt Density", "Wall Density", "Score"])
df2["Score"] = pd.to_numeric(df2["Score"])

# Line plot of Average Score vs. Dirt Density per Agent
def plot_avg_score_vs_dirt(df):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Dirt Density", y="Score", hue="Agent", marker="o")
    plt.title("Average Score vs. Dirt Density")
    plt.xlabel("Dirt Density")
    plt.ylabel("Average Score")
    plt.legend(title="Agent")
    plt.grid(True)
    plt.show()

# Line plot of Average Score vs. Wall Density per Agent
def plot_avg_score_vs_wall(df):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="Wall Density", y="Score", hue="Agent", marker="o")
    plt.title("Average Score vs. Wall Density")
    plt.xlabel("Wall Density")
    plt.ylabel("Average Score")
    plt.legend(title="Agent")
    plt.grid(True)
    plt.show()

# Variance of Score vs. Dirt Density
def plot_variance_vs_dirt(df):
    var_data = df.groupby(["Agent", "Dirt Density"])["Score"].var().reset_index()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=var_data, x="Dirt Density", y="Score", hue="Agent", marker="o")
    plt.title("Score Variance vs. Dirt Density")
    plt.xlabel("Dirt Density")
    plt.ylabel("Score Variance")
    plt.legend(title="Agent")
    plt.grid(True)
    plt.show()

# Variance of Score vs. Wall Density
def plot_variance_vs_wall(df):
    var_data = df.groupby(["Agent", "Wall Density"])["Score"].var().reset_index()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=var_data, x="Wall Density", y="Score", hue="Agent", marker="o")
    plt.title("Score Variance vs. Wall Density")
    plt.xlabel("Wall Density")
    plt.ylabel("Score Variance")
    plt.legend(title="Agent")
    plt.grid(True)
    plt.show()

# Heatmap for Battery Capacity vs. Performance
def plot_battery_vs_performance(df):
    df_summary = df.groupby(["Battery Capacity", "Dirt Density"]).agg({"Score": "mean"}).reset_index()

    df_summary["Battery Capacity"] = pd.to_numeric(df_summary["Battery Capacity"], errors="coerce")

    df_pivot = df_summary.pivot(index="Battery Capacity", columns="Dirt Density", values="Score")

    df_pivot = df_pivot.sort_index(ascending=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, annot=True, cmap="coolwarm", fmt=".1f", linewidths=0.5)

    plt.title("Battery Capacity vs. Performance")
    plt.xlabel("Dirt Density")
    plt.ylabel("Battery Capacity")

    plt.show()

plot_avg_score_vs_dirt(df1)
plot_avg_score_vs_wall(df1)
plot_variance_vs_dirt(df1)
plot_variance_vs_wall(df1)
plot_battery_vs_performance(df2)

