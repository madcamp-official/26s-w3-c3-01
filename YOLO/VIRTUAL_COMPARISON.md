# 정지 배치 비교 영상

`create_virtual_comparison.py`는 원본 방송 화면과 가상 당구대를 좌우로 배치합니다.
가상 당구대는 정면 화면에서 세 공이 완전히 정지한 경우에만 갱신됩니다.

- 리플레이 여부와 관계없이 실제 정지 상태만 사용합니다.
- 화면 전환 직전 좌표를 저장하는 `pre_cut` 처리는 사용하지 않습니다.
- 세 공이 기본 1초 동안 `0.004` 이하로만 흔들릴 때 정지로 판정합니다.
- 같은 정지 상태에서는 이벤트를 한 번만 생성합니다.
- 당구대 좌표는 색상 외곽이 아니라 쿠션 안쪽 네 선을 기준으로 합니다.

## video1 앞 10분 생성

```powershell
cd YOLO
.venv\Scripts\Activate.ps1
python create_virtual_comparison.py video1.mp4 `
  --table config\video1_table.json `
  --output-dir outputs\video1_virtual_comparison_stopped_10min `
  --sample-fps 10 `
  --max-seconds 600
```

결과 파일:

```text
outputs/video1_virtual_comparison_stopped_10min/comparison.mp4
outputs/video1_virtual_comparison_stopped_10min/events.csv
outputs/video1_virtual_comparison_stopped_10min/events.jsonl
outputs/video1_virtual_comparison_stopped_10min/event_metadata.json
outputs/video1_virtual_comparison_stopped_10min/events/
outputs/video1_virtual_comparison_stopped_10min/summary.json
```

더 엄격하거나 느슨한 정지 판정이 필요하면 `--stable-seconds`와
`--stop-threshold`를 조정합니다.
