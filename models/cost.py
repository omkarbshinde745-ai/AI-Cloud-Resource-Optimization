def cost_calc(df):
    df['cost'] = (
        df['cpu_utilization'] * 0.05 +
        df['memory_utilization'] * 0.03 +
        df['disk_usage'] * 0.02 +
        df['network_usage'] * 0.01
    )

    total_cost = df['cost'].sum()
    avg_cost = df['cost'].mean()

    return df, total_cost, avg_cost


def calculate_savings(df):
    optimized_costs = []

    for _, row in df.iterrows():
        base_cost = row['cost']

        if row['usage_type'] == "LOW":
            optimized_costs.append(base_cost * 0.7)
        elif row['usage_type'] == "HIGH":
            optimized_costs.append(base_cost * 1.1)
        else:
            optimized_costs.append(base_cost)

    df['optimized_cost'] = optimized_costs

    current_total = df['cost'].sum()
    optimized_total = df['optimized_cost'].sum()
    savings = current_total - optimized_total

    return df, current_total, optimized_total, savings