import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import requests


SOURCES = {
    "asi-readme": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2026/README.md",
    "asi-schema": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2026/docs/schema.md",
    "asi-download": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2026/docs/data_download.md",
    "alibaba-readme": "https://raw.githubusercontent.com/alibaba/clusterdata/master/README.md",
    "azure-2023": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzureLLMInferenceDataset2023.md",
    "azure-2024": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzureLLMInferenceDataset2024.md",
    "azure-multimodal": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzureLMMInferenceDataset2025.md",
    "azure-functions": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzureFunctionsDataset2019.md",
    "azure-license": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/LICENSE",
    "burstgpt": "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/README.md",
    "mooncake": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/README.md",
    "servegen": "https://raw.githubusercontent.com/alibaba/ServeGen/main/README.md",
    "servegen-offline": "https://raw.githubusercontent.com/alibaba/ServeGen/main/data/offline/README.md",
    "pai": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/README.md",
    "microservices": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-microservices-v2021/README.md",
    "maas": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2026-maas/README.md",
    "maas-schema": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2026-maas/docs/schema.md",
    "maas-download": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2026-maas/docs/data_download.md",
    "gentd26": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-v2026-GenAI/README.md",
    "google": "https://raw.githubusercontent.com/google/cluster-data/master/ClusterData2019.md",
    "google-power": "https://raw.githubusercontent.com/google/cluster-data/master/PowerData2019.md",
    "meta": "https://raw.githubusercontent.com/facebookresearch/distributed_traces/main/README.md",
    "meta-summary": "https://raw.githubusercontent.com/facebookresearch/distributed_traces/main/summary_data_atc23/README.md",
    "tracelab": "https://raw.githubusercontent.com/uw-syfi/TraceLab/main/README.md",
    "freeinference": "https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/raw/main/README.md",
    "chutes": "https://huggingface.co/datasets/harvardMadsys/chutes_metadata/raw/main/README.md",
    "chutes-code": "https://raw.githubusercontent.com/HarvardMadSys/chutes_workload/main/README.md",
}


def github(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True, encoding="utf-8"))


def sources(output):
    output.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, url = item
        response = requests.get(url, timeout=90)
        record = {"name": name, "url": url, "status": response.status_code,
                  "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest()}
        if response.ok:
            (output / f"{name}.md").write_bytes(response.content)
        return record

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    metadata = {}
    for repo, tag in [("HPMLL/BurstGPT", "v2.0"), ("Azure/AzurePublicDataset", "dataset-llm-2024"),
                      ("uw-syfi/TraceLab", "latest")]:
        endpoint = f"repos/{repo}/releases/{'latest' if tag == 'latest' else 'tags/' + tag}"
        release = github(endpoint)
        metadata[f"{repo}:{tag}"] = {"tag": release["tag_name"], "url": release["html_url"],
            "assets": [{key: a.get(key) for key in ("name", "size", "digest", "browser_download_url")}
                       for a in release["assets"]]}
    for repo in ["alibaba/clusterdata", "HPMLL/BurstGPT", "alibaba/ServeGen", "kvcache-ai/Mooncake"]:
        info = github(f"repos/{repo}")
        metadata[repo] = {key: info.get(key) for key in ("license", "default_branch", "pushed_at", "size")}
        tree = github(f"repos/{repo}/git/trees/{info['default_branch']}?recursive=1")
        metadata[repo]["tree_sha"] = tree["sha"]
        metadata[repo]["files"] = [{key: f.get(key) for key in ("path", "size", "sha")}
            for f in tree["tree"] if f["type"] == "blob" and
            ("LICENSE" in f["path"].upper() or (repo != "alibaba/clusterdata" and
             any(x in f["path"] for x in ("data/", "traces/", "arxiv-trace/"))))]
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "sources": records, "github": metadata}
    (output / "source-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def reviews(repo, number, output):
    record = {"checked_at": datetime.now(timezone.utc).isoformat(), "repo": repo, "number": number}
    for key, endpoint in [("comments", f"issues/{number}/comments"),
                          ("inline_comments", f"pulls/{number}/comments"),
                          ("reviews", f"pulls/{number}/reviews")]:
        pages = json.loads(subprocess.check_output(["gh", "api", "--paginate", "--slurp",
            f"repos/{repo}/{endpoint}"], text=True, encoding="utf-8"))
        record[key] = [item for page in pages for item in page]
    owner, name = repo.split("/")
    query = """query($owner:String!,$name:String!,$number:Int!,$after:String) {
      repository(owner:$owner,name:$name) { pullRequest(number:$number) {
        url state reviewThreads(first:100,after:$after) {
          pageInfo { hasNextPage endCursor }
          nodes { id isResolved comments(first:100) {
            pageInfo { hasNextPage endCursor }
            nodes { id body createdAt author { login } }
          } }
        }
      } }
    }"""
    threads, cursor = [], None
    while True:
        payload = {"query": query, "variables": {"owner": owner, "name": name, "number": number, "after": cursor}}
        result = json.loads(subprocess.check_output(["gh", "api", "graphql", "--input", "-"],
            input=json.dumps(payload), text=True, encoding="utf-8"))
        if "errors" in result:
            raise RuntimeError(result["errors"])
        pr = result["data"]["repository"]["pullRequest"]
        page = pr["reviewThreads"]
        for thread in page["nodes"]:
            comment_page = thread["comments"]
            thread["all_comments"] = comment_page["nodes"][:]
            while comment_page["pageInfo"]["hasNextPage"]:
                nested = {"query": """query($id:ID!,$after:String!) { node(id:$id) {
                  ... on PullRequestReviewThread { comments(first:100,after:$after) {
                    pageInfo { hasNextPage endCursor } nodes { id body createdAt author { login } }
                  } } } }""", "variables": {"id": thread["id"], "after": comment_page["pageInfo"]["endCursor"]}}
                more = json.loads(subprocess.check_output(["gh", "api", "graphql", "--input", "-"],
                    input=json.dumps(nested), text=True, encoding="utf-8"))
                if "errors" in more:
                    raise RuntimeError(more["errors"])
                comment_page = more["data"]["node"]["comments"]
                thread["all_comments"].extend(comment_page["nodes"])
        threads.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    record.update(url=pr["url"], state=pr["state"], threads=threads)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核验推理数据官方资料、文件元数据与分页 PR 评论")
    parser.add_argument("mode", choices=["sources", "reviews"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", default="TuzzyT-Laiii/33")
    parser.add_argument("--pr", type=int, default=27)
    args = parser.parse_args()
    if args.mode == "sources":
        sources(args.output)
    else:
        reviews(args.repo, args.pr, args.output)
