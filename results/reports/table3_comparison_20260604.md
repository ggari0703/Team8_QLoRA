# QLoRA Table 3 비교 리포트

작성 기준: 2026-06-04 KST

결과 파일:

- GLUE: `reproduce_table3/results/table3_glue_table3.csv`
- SNI: `reproduce_table3/results/table3_sni_local_50k.csv`
- 논문 기준값: QLoRA 논문 Table 3, "Experiments comparing 16-bit BrainFloat (BF16), 8-bit Integer (Int8), 4-bit Float (FP4), and 4-bit NormalFloat (NF4) on GLUE and Super-NaturalInstructions." 원문 PDF: <https://papers.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf>

주의: 이 파일의 결과는 현재 저장된 sanity/축소 실험 결과이다. 논문 Table 3 전체 재현 결과가 아니다.

## 실행 하드웨어

| 항목 | 값 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090 24GB x 2 |
| Driver | 570.211.01 |
| CUDA | 12.8 |
| 확인 시점 GPU 상태 | 학습 프로세스 없음, GPU util 0% |

## GLUE 비교 기준

사용자 요청에 맞춰 `metric_name=accuracy`가 기록된 GLUE 실험만 집계했다.

포함한 태스크:

- SST-2
- MRPC
- QQP
- MNLI
- QNLI
- RTE

제외한 태스크:

- CoLA: `matthews_correlation`
- STS-B: 상관계수 기반 metric

따라서 아래 평균은 "accuracy metric이 있는 6개 태스크 평균"이다. QLoRA 논문 Table 3의 GLUE 단일 값과 완전히 동일한 집계 조건이라고 해석하면 안 된다.

## GLUE 실험 세팅

| 항목 | 값 |
| --- | --- |
| task group | `roberta_glue` |
| model | `FacebookAI/roberta-large` |
| dataset | `glue/*` |
| seed | 42 |
| max length | 128 |
| train batch size | 8 |
| eval batch size | 16 |
| gradient accumulation | 4 |
| effective train batch size | 32 |
| max steps | -1, epoch 기반 |
| scheduler | linear |
| warmup ratio | 0.06 |
| max grad norm | 1.0 |
| save strategy | `no` |
| gradient checkpointing | false |
| BF16 | true |

| method | quantization | LoRA | optimizer | LoRA target modules | r / alpha / dropout |
| --- | --- | --- | --- | --- | --- |
| `bf16` | none | disabled | `adamw_torch` | - | - |
| `lora_bf16` | none | enabled | `adamw_torch` | `query`, `value` | 8 / 16 / 0.0 |
| `qlora_int8` | 8-bit int | enabled | `adamw_torch` | `query`, `value` | 8 / 16 / 0.0 |
| `qlora_fp4` | 4-bit FP4 | enabled | `adamw_torch` | `query`, `value` | 8 / 16 / 0.0 |

| GLUE task | learning rate | epochs |
| --- | ---: | ---: |
| SST-2 | 4e-4 | 10 |
| MRPC | 3e-4 | 20 |
| QQP | 3e-4 | 10 |
| MNLI | 3e-4 | 3 |
| QNLI | 2e-4 | 3 |
| RTE | 4e-4 | 20 |

## GLUE accuracy 결과

값은 CSV의 accuracy를 퍼센트로 변환한 것이다.

| task | method | accuracy (%) | peak GPU MB |
| --- | --- | ---: | ---: |
| SST-2 | `bf16` | 50.92 | 3428.27 |
| SST-2 | `lora_bf16` | 95.99 | 1380.57 |
| SST-2 | `qlora_int8` | 95.64 | 1289.88 |
| SST-2 | `qlora_fp4` | 95.07 | 978.54 |
| MRPC | `bf16` | 68.38 | 3675.33 |
| MRPC | `lora_bf16` | 90.93 | 1380.57 |
| MRPC | `qlora_int8` | 90.69 | 1289.87 |
| MRPC | `qlora_fp4` | 88.97 | 978.88 |
| QQP | `bf16` | 63.18 | 3859.56 |
| QQP | `lora_bf16` | 91.22 | 1475.95 |
| QQP | `qlora_int8` | 91.33 | 1289.88 |
| QQP | `qlora_fp4` | 91.16 | 978.54 |
| MNLI | `bf16` | 34.75 | 3875.11 |
| MNLI | `lora_bf16` | 90.45 | 1475.97 |
| MNLI | `qlora_int8` | 90.46 | 1289.88 |
| MNLI | `qlora_fp4` | 90.08 | 978.54 |
| QNLI | `bf16` | 50.54 | 3868.63 |
| QNLI | `lora_bf16` | 94.38 | 1475.96 |
| QNLI | `qlora_int8` | 94.42 | 1289.88 |
| QNLI | `qlora_fp4` | 94.14 | 978.89 |
| RTE | `bf16` | 47.29 | 3862.34 |
| RTE | `lora_bf16` | 84.84 | 1475.98 |
| RTE | `qlora_int8` | 84.12 | 1289.76 |
| RTE | `qlora_fp4` | 82.31 | 978.55 |

## GLUE 평균 및 논문 Table 3 비교

논문 Table 3의 GLUE 값은 RoBERTa-large 기준 단일 점수이다. 현재 결과는 accuracy가 있는 6개 태스크만 평균낸 값이다.

| method | 현재 accuracy 평균 (%) | 논문 Table 3 GLUE (%) | 차이 (현재 - 논문, pp) | 평균 peak GPU MB | 최대 peak GPU MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bf16` | 52.51 | 88.60 | -36.09 | 3761.54 | 3875.11 |
| `lora_bf16` | 91.30 | 88.80 | +2.50 | 1444.17 | 1475.98 |
| `qlora_int8` | 91.11 | 88.80 | +2.31 | 1289.86 | 1289.88 |
| `qlora_fp4` | 90.29 | 88.60 | +1.69 | 978.66 | 978.89 |

관찰:

- `bf16` full finetuning은 SST-2, MNLI, QNLI, RTE에서 낮은 accuracy를 기록했다. 이 행들은 학습 설정 또는 안정성 재확인이 필요하다.
- LoRA/QLoRA 계열은 accuracy 태스크 평균 기준으로 논문 Table 3의 RoBERTa-large GLUE 값보다 높게 나왔다. 단, 현재 평균은 CoLA와 STS-B를 제외한 accuracy-only 평균이므로 논문 수치와 직접 등가 비교하면 안 된다.
- GPU peak 기준으로는 `qlora_fp4`가 가장 낮았다. 평균 peak는 `lora_bf16` 1444.17 MB, `qlora_int8` 1289.86 MB, `qlora_fp4` 978.66 MB이다.

## SNI 실험 세팅

현재 SNI 결과는 full Super-NaturalInstructions 전체 학습이 아니라 local 50k 축소 실험이다.

| 항목 | 값 |
| --- | --- |
| task group | `t5_sni` |
| dataset name | `allenai/natural-instructions-local-50k` |
| local repo path | `reproduce_table3/data/natural-instructions` |
| local split | `default` |
| train samples | 50000 |
| eval samples | 1000 |
| eval instances per task | 100 |
| input encoding | `tk_instruct_def_pos_2` |
| positive examples | 2 |
| source max length | 1024 |
| target max length | 128 |
| generation max length | 64 |
| seed | 42 |
| train batch size | 16 |
| eval batch size | 1 |
| gradient accumulation | 1 |
| effective train batch size | 16 |
| epochs | 1 |
| max steps | -1, epoch 기반 |
| learning rate | 1e-5 |
| scheduler | constant |
| warmup ratio | 0.0 |
| max grad norm | 1.0 |
| gradient checkpointing | true |
| BF16 | true |
| save strategy | `no` |

| method | quantization | double quant | LoRA | optimizer | LoRA target modules | r / alpha / dropout |
| --- | --- | --- | --- | --- | --- | --- |
| `lora_bf16` | none | false | enabled | `adamw_torch` | `all-linear` | 16 / 64 / 0.0 |
| `qlora_nf4_dq` | 4-bit NF4 | true | enabled | `paged_adamw_32bit` | `all-linear` | 16 / 64 / 0.0 |

모델 매핑:

| 논문 표기 | 현재 config model |
| --- | --- |
| T5-80M | `google/t5-v1_1-small` |
| T5-250M | `google/t5-v1_1-base` |
| T5-780M | `google/t5-v1_1-large` |

## SNI 결과 및 논문 Table 3 비교

CSV의 RougeL 값은 0-1 스케일로 저장되어 있어 논문 Table 3과 비교하기 위해 x100으로 변환했다.

| model | method | 현재 RougeL (%) | 논문 Table 3 RougeL (%) | 차이 (현재 - 논문, pp) | peak GPU MB |
| --- | --- | ---: | ---: | ---: | ---: |
| T5-80M | `lora_bf16` | 5.64 | 40.50 | -34.86 | 2441.60 |
| T5-80M | `qlora_nf4_dq` | 7.01 | 40.40 | -33.39 | 3206.31 |
| T5-250M | `lora_bf16` | 10.13 | 42.60 | -32.47 | 5199.90 |
| T5-250M | `qlora_nf4_dq` | 10.44 | 42.70 | -32.26 | 6268.82 |
| T5-780M | `lora_bf16` | 20.54 | 47.10 | -26.56 | 8246.81 |
| T5-780M | `qlora_nf4_dq` | 20.09 | 47.70 | -27.61 | 9711.19 |

논문 Table 3의 SNI 기준값:

| method | T5-80M | T5-250M | T5-780M | T5-3B | T5-11B |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 40.10 | 42.10 | 48.00 | 54.30 | 62.00 |
| BF16 replication | 40.00 | 42.20 | 47.30 | 54.90 | - |
| LoRA BF16 | 40.50 | 42.60 | 47.10 | 55.40 | 60.70 |
| QLoRA Int8 | 40.40 | 42.90 | 45.40 | 56.50 | 60.70 |
| QLoRA FP4 | 40.30 | 42.40 | 47.50 | 55.60 | 60.90 |
| QLoRA NF4 + DQ | 40.40 | 42.70 | 47.70 | 55.30 | 60.90 |

관찰:

- 현재 SNI 결과는 논문보다 크게 낮다. 이는 full SNI 전체 학습이 아니라 50k train / 1k eval 축소, 1 epoch 설정이기 때문이다.
- 축소 실험 내부 비교에서는 T5-80M과 T5-250M에서 `qlora_nf4_dq`가 `lora_bf16`보다 약간 높다.
- T5-780M에서는 `lora_bf16`이 `qlora_nf4_dq`보다 약간 높다.
- peak GPU memory는 모델 크기와 함께 증가했다. T5-780M 기준 `lora_bf16` 8246.81 MB, `qlora_nf4_dq` 9711.19 MB로 기록되었다.

## 결론

- GLUE accuracy-only 평균에서는 LoRA/QLoRA 계열이 논문 Table 3 GLUE 값과 비슷하거나 높게 집계되었다. 하지만 현재 평균은 accuracy metric 태스크만 대상으로 하므로 논문 GLUE 집계와 완전히 같은 비교는 아니다.
- GLUE full `bf16` 행은 여러 태스크에서 낮게 나와 재실험 또는 hyperparameter 재검증 대상이다.
- SNI는 데이터와 학습량을 크게 줄인 sanity 실험이라 논문 Table 3 RougeL과 직접 재현 비교가 되지 않는다. 현재 결과는 구현 경로, logging, quantization, LoRA 설정 검증용으로 해석해야 한다.
