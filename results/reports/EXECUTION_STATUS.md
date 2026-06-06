# 실행 상태

작성 시점: 2026-06-01

## 현재 실행 중인 hardware-scope 실험

요청 범위인 T5-80M/250M/780M + RoBERTa-large 실제 SNI/GLUE 실험은 GPU별 tmux session으로 나누어 실행 중입니다.

- SNI/T5 tmux session: `table3_sni_gpu0`
- SNI/T5 pilot manifest: `reproduce_table3/configs/hardware_scope_sni_pilot_manifest.yaml`
- SNI/T5 pilot 로그: `reproduce_table3/logs/hardware_scope_sni_gpu0_pilot_20260601.log`
- SNI/T5 pilot 결과 CSV: `reproduce_table3/results/table3_hardware_scope_pilot.csv`
- RoBERTa/GLUE tmux session: `table3_roberta_gpu1`
- RoBERTa/GLUE manifest: `reproduce_table3/configs/hardware_scope_roberta_manifest.yaml`
- RoBERTa/GLUE 로그: `reproduce_table3/logs/hardware_scope_roberta_gpu1_rerun_20260601.log`
- 결과 CSV: `reproduce_table3/results/table3_hardware_scope.csv`

SNI/T5 첫 run은 QLoRA 기본값에 가까운 `learning_rate: 2e-4`, `max_steps: 10000` 및 definition-only 입력으로 시작되었지만, QLoRA Appendix A.2와 Wang et al. [60]의 Tk-Instruct 설정과 맞지 않아 중단했습니다. 이후 입력 인코딩을 definition + two positive examples로 바꾸고, T5/SNI 학습 설정을 `learning_rate: 1e-5`, constant scheduler, 2 epoch, effective batch size 16, source length 1024, target length 128로 수정했습니다.

다만 실제 full SNI는 train example 약 306만 개라 T5-80M BF16 한 개만 2 epoch 기준 약 55시간이 걸리는 것으로 확인되었습니다. full SNI 설정은 보존하고, 현재는 실제 SNI 데이터의 pilot subset으로 `max_train_samples: 16000`, `max_eval_samples: 1000`, `max_steps: 1000`을 적용한 6개 실험(T5-80M/250M/780M x LoRA BF16/QLoRA NF4+DQ)을 실행 중입니다. 이 pilot 결과는 full Table 3 재현 숫자가 아니라 하드웨어 범위에서 비교가 가능한 실제 데이터 run입니다.

RoBERTa-large 첫 BF16 run은 `learning_rate: 2e-4`, `lr_scheduler_type: constant`, `max_steps: 10000` 조합으로 loss가 0.69대에 머물렀고 SST-2 accuracy가 0.509로 기록되었습니다. 이 row는 실패한 탐색 run으로 남겨두고 삭제하지 않았습니다. 이후 RoBERTa GLUE 설정을 `learning_rate: 2e-5`(full BF16) / `1e-4`(LoRA/QLoRA), linear scheduler, warmup 6%, 3 epoch, effective batch size 32로 수정해 재시작했습니다.

상태 확인:

```bash
tmux capture-pane -pt table3_sni_gpu0 -S -120
tmux capture-pane -pt table3_roberta_gpu1 -S -120
tail -n 100 reproduce_table3/logs/hardware_scope_sni_gpu0_pilot_20260601.log
tail -n 100 reproduce_table3/logs/hardware_scope_roberta_gpu1_rerun_20260601.log
nvidia-smi
```

세션에 attach:

```bash
tmux attach -t table3_sni_gpu0
tmux attach -t table3_roberta_gpu1
```

중단:

```bash
tmux send-keys -t table3_sni_gpu0 C-c
tmux send-keys -t table3_roberta_gpu1 C-c
```

## 완료된 sanity 실험

다음 실험은 실제로 실행되어 `reproduce_table3/results/table3_runs.csv`에 기록되었습니다.

| task_group | model | method | dataset | metric | value | peak GPU MB |
| --- | --- | --- | --- | --- | --- | --- |
| t5_sni | t5-small | lora_bf16 | toy_sni | RougeL | 0.0 | 381.85 |
| t5_sni | t5-small | qlora_nf4_dq | toy_sni | RougeL | 0.0 | 127.49 |
| roberta_glue | FacebookAI/roberta-base | lora_bf16 | glue/sst2 | Accuracy | 0.515625 | 772.02 |
| roberta_glue | FacebookAI/roberta-base | qlora_fp4 | glue/sst2 | Accuracy | 0.515625 | 269.35 |

이 결과는 코드 경로 검증용입니다. toy SNI와 10 step GLUE debug run이므로 논문 Table 3 숫자와 비교하면 안 됩니다.

## 생성된 full Table 3 config

`python3 -m reproduce_table3.experiment_files.config_generators.make_table3_configs`로 35개 설정을 생성했습니다.

- T5: 5개 모델 크기 x 6개 방법 = 30개
- GLUE/SST-2: RoBERTa-large x 5개 방법 = 5개
- manifest: `reproduce_table3/configs/table3_full_manifest.yaml`

전체 실행 명령:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --manifest reproduce_table3/configs/table3_full_manifest.yaml \
  --continue-on-error
```

## 이번 세션에서 실패하거나 중단한 명령

- `python3 -m reproduce_table3.experiment_files.runner.run_experiments --config reproduce_table3/configs/t5_lora_bf16.yaml`
  - 초기 설정이 Hugging Face의 SNI 전체 task 파일을 다운로드하려고 해서 중단했습니다.
  - 조치: 작은 sanity config는 `toy_sni`로 고정했습니다. full config는 별도 `table3_full` 아래에 남겼습니다.

- `python3 -m reproduce_table3.experiment_files.runner.run_experiments --config reproduce_table3/configs/roberta_qlora_fp4.yaml`
  - 초기 구현은 `AutoModelForSequenceClassification` + 4bit quantization 조합에서 missing classifier head 초기화가 실패했습니다.
  - 조치: quantized RoBERTa backbone + 별도 classification head 래퍼로 수정했습니다.

- 수정 후 RoBERTa QLoRA FP4 첫 재시도
  - 2GPU DataParallel이 bitsandbytes 4bit quant_state를 cuda:0/cuda:1에 나누어 실패했습니다.
  - 조치: quantized 실행은 기본 `device_map: {"": 0}`로 고정하고 wrapper를 model-parallel로 표시해 Trainer DataParallel을 피했습니다.

## 현재 하드웨어 한계

현재 확인된 GPU는 RTX 3090 24GB 2장입니다. 이 환경으로 RoBERTa-large LoRA/QLoRA와 T5-small/base/large 일부 QLoRA 실험은 시도할 수 있지만, 원 Table 3 전체를 그대로 실행하기에는 특히 다음 항목이 병목입니다.

- T5-11B BF16 full finetuning
- T5-11B BF16 replication
- T5-3B 이상 full BF16 계열
- full Super-NaturalInstructions 다운로드, 전처리, 전체 step 학습
- 여러 seed 반복과 평균/분산 산출

따라서 현재 산출물은 “전체 재현 실행 결과”가 아니라, full matrix를 실행할 수 있게 만든 설정/코드와 작은 검증 결과입니다.
