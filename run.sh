#!/bin/sh
# MAT one-command runner: kiểm tra môi trường, chạy Docker đúng kịch bản,
# chờ API khỏe rồi gọi thử — xong là dự án chạy hoàn chỉnh.
#
#   ./run.sh            (hỏi: 1 = baseline CPU, 2 = transformer từ Hub)
set -eu

cd "$(dirname "$0")"
COMPOSE="docker compose -f infra/docker-compose.yml"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Thiếu $1 — cài trước rồi chạy lại."
    exit 1
  fi
}

wait_healthy() {
  for _ in $(seq 1 150); do
    if curl --fail --silent http://127.0.0.1:8000/health-check >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

sample_calls() {
  echo "--- Gọi thử /predict ---"
  curl --fail --silent -H 'Content-Type: application/json' \
    -d '{"text":"Mazda CX-5 chạy êm, tăng tốc mượt"}' \
    http://127.0.0.1:8000/predict
  echo
  curl --fail --silent -H 'Content-Type: application/json' \
    -d '{"text":"Tuyệt vời, xe mới một tuần đã nằm xưởng hai lần."}' \
    http://127.0.0.1:8000/predict
  echo
}

finish() {
  SVC="$1"
  echo "=== XONG ==="
  echo "API : http://127.0.0.1:8000  (docs: /docs nếu mở)"
  echo "Web : http://127.0.0.1:3000"
  echo "Log : $COMPOSE logs -f $SVC"
  echo "Tắt : $COMPOSE down"
}

# Chỉ build image còn thiếu; có sẵn thì up luôn cho nhanh.
# Trả về "--build" khi thiếu ít nhất 1 image, ngược lại chuỗi rỗng.
build_flag() {
  for image in "$@"; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      echo "--build"
      return 0
    fi
  done
  echo ""
}

need docker
need curl
if [ ! -f ai/data/dataset.csv ]; then
  echo "Thiếu ai/data/dataset.csv — đặt dataset vào đó trước."
  exit 1
fi

# Chặn build khi đĩa sắp đầy.
require_disk_gb() {
  free_gb=$(df -BG . | tail -n 1 | awk '{print $4}' | tr -d 'G')
  if [ "$free_gb" -lt "$1" ]; then
    echo "Đĩa còn ${free_gb} GB, cần ít nhất $1 GB. Dọn bớt (docker system prune) rồi chạy lại."
    exit 1
  fi
}

echo "Chọn chế độ chạy:"
echo "  1) Baseline — phục vụ mô hình có sẵn trên CPU"
echo "  2) Transformer — tải BamiBERT đã train từ Hub rồi phục vụ trên CPU"
printf "Nhập 1 hoặc 2: "
read -r choice

case "$choice" in
  1)
    # Số đo thật: image CPU ~2.0 GB (torch CPU) + web ~0.9 GB + dư địa build.
    require_disk_gb 6
    # shellcheck disable=SC2086
    $COMPOSE up $(build_flag infra-ai:latest infra-web:latest) -d ai web
    if wait_healthy; then
      sample_calls
      finish ai
    else
      echo "API chưa khỏe sau 5 phút. Xem log: $COMPOSE logs ai"
      exit 1
    fi
    ;;
  2)
    # Số đo thật: image ~2.0 GB + web ~0.9 GB + artifact Hub ~0.4 GB vào volume.
    require_disk_gb 8
    # shellcheck disable=SC2086
    $COMPOSE --profile trans up $(build_flag infra-ai-trans:latest infra-web:latest) -d ai-trans web
    if wait_healthy; then
      sample_calls
      finish ai-trans
    else
      echo "API chưa khỏe sau 5 phút. Xem log: $COMPOSE logs ai-trans"
      exit 1
    fi
    ;;
  *)
    echo "Lựa chọn không hợp lệ (chỉ 1 hoặc 2)."
    exit 1
    ;;
esac
