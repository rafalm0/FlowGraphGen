import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# def dig(repo):
#     if ".py" in repo:
#         return repo
#     else:
#         return dig(repo + "\\" + os.listdir(repo)[0])
#
#
# headers = {}
# for instance_id in os.listdir("data"):
#     if instance_id == "metadata.csv":
#         continue
#     repo_file = dig(f"data\\{instance_id}\\original")
#     repo_path = repo_file.replace(f"data\\{instance_id}\\original\\", "").replace("\\", '/')
#     header = f"diff --git a/{repo_path} b/{repo_path}\n--- a/{repo_path}\n+++ b/{repo_path}\n@@"
#     headers[instance_id] = header
#
# df = pd.read_json("baseline_predictions3.jsonl", lines=True, orient='records')
# df['model_patch'] = df['model_patch'].apply(lambda a: "@@".join(a.split("@@")[1:]))
# df['model_patch'] = df.apply(lambda a: headers[a['instance_id'].replace("__", "_")] + a['model_patch'], axis=1)
# df.to_json("baseline_predictions3_processed.jsonl", orient='records', lines=True)
# print()
# pass

#
# labels = ['baseline', 'call graph']
#
# baseline = [1, 0, 1, 1]
# graph = [2, 1, 1, 1]
#
# data = [baseline, graph]
# means = [np.mean(d) for d in data]
# std_err = [np.std(d, ddof=1) / np.sqrt(len(d)) for d in data]
# ci = [1.96 * se for se in std_err]
#
# plt.figure()
# plt.bar(labels, means, yerr=ci, capsize=5)
# plt.ylabel('Score')
# plt.title('Mean with 95% Confidence Interval')
# plt.ylim(0, 20)
#
# plt.show()

