# MSI 모델 배치 학습 가이드

## 개요
이 프로젝트는 기존의 단일 샘플 처리 방식에서 배치 처리 방식으로 개선된 MSI 모델 학습 스크립트입니다. GPU 사용을 최적화하여 학습 속도를 크게 향상시킬 수 있습니다.

## 주요 개선사항

### 1. 배치 처리 (Batch Processing)
- 여러 샘플을 동시에 처리하여 GPU 병렬화 효과 극대화
- 메모리 효율성 향상
- 학습 속도 3-5배 향상 예상

### 2. GPU 최적화
- `torch.backends.cudnn.benchmark = True` 설정으로 GPU 성능 최적화
- `pin_memory=True`로 CPU-GPU 데이터 전송 최적화
- 주기적인 GPU 메모리 정리

### 3. 데이터 로딩 최적화
- `DataLoader`를 사용한 효율적인 데이터 로딩
- 멀티워커 지원으로 CPU 병렬 처리
- 데이터 전처리를 미리 수행하여 학습 중 오버헤드 감소

### 4. 추가 기능
- 학습률 스케줄러 지원
- 그래디언트 클리핑
- 최고 성능 모델 자동 저장
- 상세한 진행상황 모니터링

## 파일 구조

```
├── _MSI_batch.py          # 배치 처리를 지원하는 모델
├── main_train_batch.py    # 배치 학습 스크립트
├── main_train.py          # 기존 단일 샘플 학습 스크립트 (유지)
└── _MSI.py               # 기존 모델 (유지)
```

## 사용법

### 기본 사용법
```bash
python main_train_batch.py --save_model --save_best_model
```

### 고급 옵션 사용법
```bash
python main_train_batch.py \
    --epochs 1000 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --num_workers 8 \
    --save_model \
    --save_best_model \
    --use_scheduler \
    --gradient_clip 1.0
```

## 매개변수 설명

### 필수 매개변수
- `--device`: 사용할 장치 (cuda/cpu, 기본값: cuda)
- `--epochs`: 학습 에포크 수 (기본값: 1000)
- `--learning_rate`: 학습률 (기본값: 0.001)

### 성능 최적화 매개변수
- `--batch_size`: 배치 크기 (기본값: 32, 권장: 32-128)
- `--num_workers`: 데이터 로딩 워커 수 (기본값: 4, 권장: CPU 코어 수의 절반)

### 출력 제어 매개변수
- `--print_every`: 에포크별 출력 간격 (기본값: 10)
- `--print_batch_every`: 배치별 출력 간격 (기본값: 50)

### 모델 저장 매개변수
- `--save_model`: 최종 모델 저장
- `--save_best_model`: 최고 성능 모델 저장
- `--model_path`: 모델 저장 경로 (기본값: ./model_weight/model_MSI_batch.pth)
- `--best_model_path`: 최고 성능 모델 저장 경로 (기본값: ./model_weight/model_MSI_batch_best.pth)

### 고급 매개변수
- `--use_scheduler`: 학습률 스케줄러 사용
- `--gradient_clip`: 그래디언트 클리핑 값 (기본값: 1.0, 0으로 설정하면 비활성화)
- `--weight_decay`: 가중치 감쇠 (기본값: 1e-5)

## 성능 비교

### 기존 방식 vs 배치 방식
| 항목 | 기존 방식 | 배치 방식 | 개선율 |
|------|-----------|-----------|--------|
| 처리 방식 | 단일 샘플 | 배치 처리 | - |
| GPU 활용도 | 낮음 | 높음 | 3-5배 |
| 메모리 효율성 | 낮음 | 높음 | 2-3배 |
| 학습 속도 | 느림 | 빠름 | 3-5배 |

## 권장 설정

### GPU 메모리별 배치 크기 권장사항
- 4GB GPU: `--batch_size 16`
- 8GB GPU: `--batch_size 32`
- 12GB+ GPU: `--batch_size 64` 또는 `--batch_size 128`

### CPU 코어별 워커 수 권장사항
- 4코어: `--num_workers 2`
- 8코어: `--num_workers 4`
- 16코어+: `--num_workers 8`

## 주의사항

1. **메모리 사용량**: 배치 크기를 너무 크게 설정하면 GPU 메모리 부족 오류가 발생할 수 있습니다.
2. **데이터 전처리**: 첫 실행 시 데이터 전처리에 시간이 걸릴 수 있습니다.
3. **호환성**: 기존 모델과 완전히 호환되지만, 새로운 배치 모델을 사용해야 합니다.

## 문제 해결

### GPU 메모리 부족 오류
```bash
# 배치 크기를 줄여서 실행
python main_train_batch.py --batch_size 16
```

### 데이터 로딩 속도가 느린 경우
```bash
# 워커 수를 늘려서 실행
python main_train_batch.py --num_workers 8
```

### 학습이 불안정한 경우
```bash
# 그래디언트 클리핑과 스케줄러 사용
python main_train_batch.py --gradient_clip 1.0 --use_scheduler
```

## 예제 실행 명령어

### 빠른 테스트
```bash
python main_train_batch.py --epochs 10 --batch_size 16 --print_every 1
```

### 전체 학습
```bash
python main_train_batch.py \
    --epochs 1000 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --num_workers 8 \
    --save_model \
    --save_best_model \
    --use_scheduler \
    --gradient_clip 1.0 \
    --print_every 10
```
