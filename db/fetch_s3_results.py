# S3에 모인 결과(turns.jsonl / turns.csv)를 로컬로 내려받아 모은다.
# 캠퍼스망에서 RDS(5432)는 막혀도 S3(443)는 되므로, DB 없이 데이터를 바로 확인할 때 유용.
#
# 사용법 (db/db.env 에 S3_BUCKET, AWS_REGION 이 있으면 그대로 읽음):
#   source db/db.env && venv/bin/python fetch_s3_results.py
#   venv/bin/python fetch_s3_results.py --outdir results_s3 --all-files
#
# 결과:
#   results_s3/<video_id>/turns.jsonl(+csv)   ← 영상별 원본
#   results_s3/all_turns.jsonl                ← 전 영상 턴을 한 파일로 합친 것(보기 편함)
import argparse
import json
import os

import boto3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET", ""))
    ap.add_argument("--prefix", default=os.environ.get("S3_PREFIX", "results").strip("/"))
    ap.add_argument("--outdir", default="results_s3")
    ap.add_argument("--all-files", action="store_true",
                    help="turns.jsonl/csv 뿐 아니라 qa 프레임 등 모든 파일 내려받기")
    args = ap.parse_args()
    if not args.bucket:
        raise SystemExit("S3_BUCKET 이 필요합니다 (source db/db.env 후 실행하거나 --bucket).")
    bucket = args.bucket[len("s3://"):] if args.bucket.startswith("s3://") else args.bucket

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or None)
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{args.prefix}/"):
        for o in page.get("Contents", []):
            keys.append(o["Key"])

    def want(k):
        if args.all_files:
            return True
        return k.endswith("turns.jsonl") or k.endswith("turns.csv")

    targets = [k for k in keys if want(k)]
    os.makedirs(args.outdir, exist_ok=True)
    n = 0
    for k in targets:
        rel = k[len(args.prefix) + 1:] if k.startswith(args.prefix + "/") else k
        dst = os.path.join(args.outdir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(bucket, k, dst)
        n += 1
    print(f"내려받음: {n}개 파일 → {args.outdir}/")

    # 전 영상 turns.jsonl 을 하나로 합치기
    combined = os.path.join(args.outdir, "all_turns.jsonl")
    total = 0
    with open(combined, "w") as out:
        for root, _, files in sorted(os.walk(args.outdir)):
            if "turns.jsonl" in files:
                for line in open(os.path.join(root, "turns.jsonl")):
                    if line.strip():
                        out.write(line)
                        total += 1
    print(f"합본: {combined} ({total}턴)")


if __name__ == "__main__":
    main()
