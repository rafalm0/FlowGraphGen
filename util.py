import os
import pandas as pd
import matplotlib.pyplot as plt

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





