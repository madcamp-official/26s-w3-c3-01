# 정지 배치 버츄얼 비교 영상

`create_virtual_comparison.py`는 고정 천장 뷰에서 세 공이 모두 멈춘 순간의
좌표만 확정합니다. 원본 방송은 왼쪽, 마지막으로 확정된 버츄얼 공 배치는
오른쪽에 표시됩니다.

공이 움직이는 동안과 다른 카메라 화면에서는 버츄얼 배치를 변경하지 않습니다.
완전 정지 확인 전에 리플레이로 전환되면, 마지막 3개 검출의 이동량이 충분히
작은 경우 화면 전환 직전 좌표를 `pre_cut` 이벤트로 먼저 확정합니다. 리플레이
후 같은 배치가 다시 검출되어도 중복 이벤트로 저장하지 않습니다.

## video1 앞 10분 생성

```powershell
cd YOLO
.venv\Scripts\Activate.ps1
python create_virtual_comparison.py video1.mp4 `
  --table config\video1_table.json `
  --output-dir outputs\video1_virtual_comparison_10min `
  --sample-fps 10 `
  --max-seconds 600
```

결과 파일:

```text
outputs/video1_virtual_comparison_10min/comparison.mp4
outputs/video1_virtual_comparison_10min/events.csv
outputs/video1_virtual_comparison_10min/events.jsonl
outputs/video1_virtual_comparison_10min/events/
outputs/video1_virtual_comparison_10min/summary.json
```

실행 중 화면도 함께 보려면 `--show`를 추가합니다.
