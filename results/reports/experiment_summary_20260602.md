# QLoRA Table 3 재현 실험 중간 결과 정리

작성 기준: 2026-06-02 17:50:50 KST  
결과 출처:

- `results/table3_sni_local_50k.csv`
- `results/table3_glue_table3.csv`
- `configs/sni_local_50k_manifest.yaml`
- `configs/glue_table3_manifest.yaml`
- `logs/sni_local_50k_gpu0_20260602.log`
- `logs/glue_table3_gpu1_20260602.log`

이 문서는 현재까지 완료되어 CSV에 기록된 실험 결과와, 로그상 진행 중인 실험 상태를 정리한다. 아직 QLoRA 논문 Table 3 전체 재현은 완료되지 않았으며, 아래 결과는 현재 설정에서의 중간 산출물이다.

## 전체 진행 상태

| 실험 묶음 | Manifest | 완료/전체 | 현재 상태 |
| --- | --- | ---: | --- |
| T5 SNI local 50k | `configs/sni_local_50k_manifest.yaml` | 4/6 | `t5_780m_lora_bf16` 실행 중 |
| RoBERTa GLUE Table 3 scope | `configs/glue_table3_manifest.yaml` | 14/32 | `qqp_qlora_int8` 실행 중 |

로그 기준 진행 중 항목:

| 실험 | 진행률 | 진행 단계 |
| --- | ---: | --- |
| `sni_local_50k/t5_780m_lora_bf16` | 1365/3125, 약 44% | 학습 중 |
| `glue_table3/qqp_qlora_int8` | 8972/113710, 약 8% | 학습 중 |

## 공통 실행 방식

| 항목 | 값 |
| --- | --- |
| 실행 단위 | YAML config를 manifest 순서대로 실행 |
| 결과 저장 | `results/*.csv` |
| 출력 디렉터리 | `outputs/...` |
| seed | 42 |
| 저장 전략 | `save_strategy: no` |
| max steps | `-1`, epoch 기반 학습 |
| dtype | BF16 사용 |
| max grad norm | 1.0 |

## T5 SNI local 50k 설정

| 항목 | 값 |
| --- | --- |
| task group | `t5_sni` |
| dataset | `allenai/natural-instructions-local-50k` |
| local repo path | `data/natural-instructions` |
| split | `default` |
| train samples | 50000 |
| eval samples | 1000 |
| input encoding | `tk_instruct_def_pos_2` |
| positive examples | 2 |
| source max length | 1024 |
| target max length | 128 |
| generation max length | 64 |
| train batch size | 16 |
| eval batch size | 1 |
| gradient accumulation | 1 |
| epochs | 1 |
| scheduler | `constant` |
| warmup ratio | 0.0 |
| LoRA target modules | `all-linear` |
| LoRA r / alpha / dropout | 16 / 64 / 0.0 |
| LoRA BF16 optimizer | `adamw_torch` |
| QLoRA NF4 DQ optimizer | `paged_adamw_32bit` |
| QLoRA quantization | 4-bit NF4, double quantization enabled |

### T5 SNI 완료 결과

| model | method | quant type | double quant | lr | batch | grad acc | metric | value | peak GPU MB |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `google/t5-v1_1-small` | `lora_bf16` | none | false | 1e-5 | 16 | 1 | RougeL | 0.0564 | 2441.60 |
| `google/t5-v1_1-small` | `qlora_nf4_dq` | nf4 | true | 1e-5 | 16 | 1 | RougeL | 0.0701 | 3206.31 |
| `google/t5-v1_1-base` | `lora_bf16` | none | false | 1e-5 | 16 | 1 | RougeL | 0.1013 | 5199.90 |
| `google/t5-v1_1-base` | `qlora_nf4_dq` | nf4 | true | 1e-5 | 16 | 1 | RougeL | 0.1044 | 6268.82 |

### T5 SNI 남은 항목

| 순서 | config | 상태 |
| ---: | --- | --- |
| 5 | `configs/sni_local_50k/t5_780m_lora_bf16.yaml` | 실행 중 |
| 6 | `configs/sni_local_50k/t5_780m_qlora_nf4_dq.yaml` | 대기 |

## RoBERTa GLUE 설정

| 항목 | 값 |
| --- | --- |
| task group | `roberta_glue` |
| model | `FacebookAI/roberta-large` |
| max length | 128 |
| train batch size | 8 |
| eval batch size | 16 |
| gradient accumulation | 4 |
| scheduler | `linear` |
| warmup ratio | 0.06 |
| BF16 optimizer | `adamw_torch` |
| LoRA target modules | `query`, `value` |
| LoRA r / alpha / dropout | 8 / 16 / 0.0 |
| QLoRA int8 | 8-bit load, LoRA enabled |
| QLoRA fp4 | 4-bit FP4 load, LoRA enabled |
| double quantization | false |

태스크별 학습 설정:

| GLUE task | learning rate | epochs |
| --- | ---: | ---: |
| CoLA | 2e-4 | 20 |
| SST-2 | 4e-4 | 10 |
| MRPC | 3e-4 | 20 |
| QQP | 3e-4 | 10 |
| STS-B | 2e-4 | 20 |
| MNLI | 3e-4 | 3 |
| QNLI | 2e-4 | 3 |
| RTE | 4e-4 | 20 |

### GLUE 완료 결과

| task | method | metric | value | glue_score | peak GPU MB |
| --- | --- | --- | ---: | ---: | ---: |
| CoLA | `bf16` | matthews_correlation | 0.0000 | 0.0000 | 3430.00 |
| CoLA | `lora_bf16` | matthews_correlation | 0.6455 | 0.6455 | 1000.95 |
| CoLA | `qlora_int8` | matthews_correlation | 0.6505 | 0.6505 | 1288.75 |
| CoLA | `qlora_fp4` | matthews_correlation | 0.6406 | 0.6406 | 977.54 |
| SST-2 | `bf16` | accuracy | 0.5092 | 0.5092 | 3428.27 |
| SST-2 | `lora_bf16` | accuracy | 0.9599 | 0.9599 | 1380.57 |
| SST-2 | `qlora_int8` | accuracy | 0.9564 | 0.9564 | 1289.88 |
| SST-2 | `qlora_fp4` | accuracy | 0.9507 | 0.9507 | 978.54 |
| MRPC | `bf16` | accuracy / f1 | 0.6838 / 0.8122 | 0.7480 | 3675.33 |
| MRPC | `lora_bf16` | accuracy / f1 | 0.9093 / 0.9343 | 0.9218 | 1380.57 |
| MRPC | `qlora_int8` | accuracy / f1 | 0.9069 / 0.9321 | 0.9195 | 1289.87 |
| MRPC | `qlora_fp4` | accuracy / f1 | 0.8897 / 0.9201 | 0.9049 | 978.88 |
| QQP | `bf16` | accuracy / f1 | 0.6318 / 0.0000 | 0.3159 | 3859.56 |
| QQP | `lora_bf16` | accuracy / f1 | 0.9122 / 0.8833 | 0.8978 | 1475.95 |

### GLUE 진행 및 남은 항목

| 순서 | config | 상태 |
| ---: | --- | --- |
| 15 | `configs/glue_table3/qqp_qlora_int8.yaml` | 실행 중 |
| 16 | `configs/glue_table3/qqp_qlora_fp4.yaml` | 대기 |
| 17-20 | `configs/glue_table3/stsb_*.yaml` | 대기 |
| 21-24 | `configs/glue_table3/mnli_*.yaml` | 대기 |
| 25-28 | `configs/glue_table3/qnli_*.yaml` | 대기 |
| 29-32 | `configs/glue_table3/rte_*.yaml` | 대기 |

## 관찰 사항

- T5 SNI local 50k에서는 현재까지 80M, 250M 모두 `qlora_nf4_dq`가 같은 크기의 `lora_bf16`보다 RougeL이 조금 높게 기록되었다.
- T5 250M은 80M보다 RougeL이 높지만, 절대 점수는 아직 sanity-check 수준이다.
- GLUE에서 full BF16 baseline은 CoLA, SST-2, QQP에서 낮게 나왔고, LoRA/QLoRA 계열이 훨씬 높은 점수를 보였다.
- GLUE CoLA에서는 `qlora_int8`이 현재까지 가장 높은 matthews correlation을 기록했다.
- GLUE SST-2와 MRPC에서는 `lora_bf16`이 현재까지 가장 높은 `glue_score`를 기록했다.
- QQP의 full BF16 baseline은 f1이 0.0으로 기록되어 있어 예측 분포 또는 학습 안정성 확인이 필요하다.

## 주의 사항

- 이 문서는 현재까지 완료된 CSV 기록과 로그 스냅샷 기준이다.
- 실행 중인 항목은 완료 후 평가가 끝나야 CSV에 추가된다.
- 아직 모든 model size, method, GLUE task가 끝난 것이 아니므로 Table 3 전체 재현 결과로 해석하면 안 된다.
- 결과 비교 시 현재 config의 batch size, epoch, dataset subset, quantization backend 조건을 함께 보고 해석해야 한다.
