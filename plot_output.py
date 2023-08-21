import sys
import matplotlib.pyplot as plt

def parse_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    datasets = []
    current_dataset = None
    current_M = None
    current_ef = None

    for line in lines:
        line = line.strip()
        
        if 'dimensions' in line:
            if current_dataset:
                datasets.append(current_dataset)
            base_count = int(line.split(":")[1].split("base")[0].strip())
            query_count = int(line.split("and")[1].split("query")[0].strip())
            current_dataset = {
                "title": line.split(":")[0].split("/")[-1].split(".")[0],
                "base_count": base_count,
                "query_count": query_count,
                "queries": []
            }
        elif line.startswith("HNSW"):
            if "build in" in line:
                current_M = int(line.split("M=")[1].split()[0])
                current_ef = int(line.split("ef=")[1].split()[0])
            else:
                OQ = int(line.split("top 100/")[1].split()[0])
                recall = float(line.split("recall ")[1].split(",")[0])
                query_time = float(line.split("query ")[1].split("s.")[0])
                throughput = (2 * current_dataset["query_count"]) / query_time
                
                current_dataset["queries"].append({
                    "OQ": OQ,
                    "recall": recall,
                    "throughput": throughput,
                    "M": current_M,
                    "ef": current_ef
                })

    datasets.append(current_dataset)
    return datasets

def is_pareto_optimal(current_point, other_points):
    for other_point in other_points:
        if other_point["recall"] >= current_point["recall"] and other_point["throughput"] >= current_point["throughput"]:
            if other_point["recall"] > current_point["recall"] or other_point["throughput"] > current_point["throughput"]:
                return False
    return True

def plot_dataset(dataset, save_as_png=False):
    plt.figure(figsize=(12, 8))
    
    recalls = [query["recall"] for query in dataset["pareto_queries"]]
    throughputs = [query["throughput"] for query in dataset["pareto_queries"]]
    annotations = [f"OQ={query['OQ']} M={query['M']} ef={query['ef']}" for query in dataset["pareto_queries"]]

    plt.scatter(recalls, throughputs, marker='o', c='blue')
    for i, txt in enumerate(annotations):
        plt.annotate(txt, (recalls[i], throughputs[i]), fontsize=9, ha='right')

    plt.xlabel("Recall")
    plt.ylabel("Throughput")
    plt.title(dataset["title"])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_as_png:
        fname = dataset["title"] + ".png"
        print("saved " + fname)
        plt.savefig(fname)
    else:
        plt.show()

def main():
    filename = sys.argv[1]
    datasets = parse_file(filename)
    for dataset in datasets:
        dataset["pareto_queries"] = [query for query in dataset["queries"] if is_pareto_optimal(query, dataset["queries"])]
        plot_dataset(dataset, save_as_png=True)

if __name__ == "__main__":
    main()
