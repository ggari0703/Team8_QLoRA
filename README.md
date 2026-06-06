# QLoRA Table 3 재현 프레임워크

이 디렉터리는 QLoRA 논문의 Table 3을 바로 전부 재현하기 위한 코드가 아니라, 작은 규모의 sanity check부터 시작하는 재현 프레임워크입니다.

Table 3의 핵심 질문은 4bit/8bit로 양자화한 모델에 LoRA를 붙여 미세조정하는 QLoRA가 16-bit LoRA 또는 full finetuning 성능에 근접할 수 있는지입니다. 원 논문은 GLUE에서는 RoBERTa-large의 Accuracy를, Super-NaturalInstructions에서는 여러 크기의 T5 모델 RougeL을 비교합니다.

## 왜 T5-small/base부터 시작하나

원 Table 3에는 T5-3B, T5-11B 같은 큰 모델이 포함됩니다. 이 모델들은 GPU 메모리, 학습 시간, 데이터 전처리 비용이 크기 때문에 처음부터 실행하면 코드 오류와 자원 문제를 구분하기 어렵습니다.

따라서 이 프레임워크의 첫 목표는 다음 조합이 정상 동작하는지 확인하는 것입니다.

- T5-small 또는 T5-base
- Super-NaturalInstructions 일부 샘플 또는 fallback toy instruction 데이터
- LoRA BF16
- QLoRA NF4 + double quantization
- RougeL 평가
- CSV 결과 저장

## 설치

저장소 루트에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r reproduce_table3/requirements.txt
```

QLoRA 4bit/8bit 실행에는 CUDA 환경에서 동작하는 `bitsandbytes`가 필요합니다. CPU만 있는 환경에서는 BF16 LoRA 설정으로 코드 경로를 먼저 확인하고, QLoRA는 CUDA GPU에서 실행해야 합니다.

SNI 실험 데이터는 이 저장소에 포함하지 않습니다. SNI 관련 실험을 실행하려면 AllenAI Natural Instructions 저장소를 `reproduce_table3/data/natural-instructions` 경로에 별도로 clone해야 합니다.

```bash
git clone --depth 1 https://github.com/allenai/natural-instructions.git \
  reproduce_table3/data/natural-instructions
```

clone 후 다음 파일과 디렉터리가 존재해야 합니다.

```text
reproduce_table3/data/natural-instructions/tasks/
reproduce_table3/data/natural-instructions/splits/default/
```

GLUE 실험에는 위 로컬 SNI 데이터가 필요하지 않습니다.

## 실제 결과 생성 환경

이 저장소의 현재 CSV 결과는 2026년 6월에 다음 환경에서 생성했습니다.

| 항목 | 실제 실행 환경 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, Linux kernel 6.8.0-106-generic, x86_64 |
| Python | `/usr/bin/python3`, Python 3.10.12 |
| Python 패키지 위치 | 주로 `/home/gyuseong/.local/lib/python3.10/site-packages` |
| GPU | NVIDIA GeForce RTX 3090 24GB 2장 |
| NVIDIA driver | 570.211.01 |
| `nvidia-smi` 표시 CUDA | 12.8 |
| 시스템 CUDA toolkit | CUDA 12.8, `nvcc` 12.8.93 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN | 9.10.2 |
| NCCL | 2.27.5 |

`nvidia-smi`의 CUDA 버전은 드라이버가 지원하는 최대 CUDA 호환 버전이고, `nvcc`는 설치된 시스템 toolkit 버전이며, `torch.version.cuda`는 PyTorch wheel이 사용하는 CUDA runtime 버전입니다. 이 환경에서는 세 값이 모두 CUDA 12.8 계열입니다.

실험은 별도 virtual environment가 아니라 `/usr/bin/python3`와 사용자 site-packages 조합으로 실행했습니다. 새 환경에서는 패키지 충돌을 피하기 위해 virtual environment 사용을 권장합니다.

현재 셸의 기본 `python3`가 Anaconda 등 다른 인터프리터를 가리킬 수 있으므로, 실행 전 다음 명령으로 Python과 패키지 경로를 확인해야 합니다. 서로 다른 Python의 `pip`와 실행기를 혼용하면 필요한 패키지를 찾지 못할 수 있습니다.

```bash
which python3
python3 --version
python3 -m pip --version
```

### 실제 사용 패키지 버전

현재 `requirements.txt`에는 패키지 이름만 있고 버전이 고정되어 있지 않습니다. 아래 표는 현재 결과를 생성할 때 실제 설치되어 있던 버전입니다.

| 패키지 | 실제 버전 | 실험에서의 역할 |
| --- | ---: | --- |
| `torch` | 2.10.0+cu128 | BF16 학습, CUDA 실행, 모델 연산 |
| `transformers` | 4.57.6 | RoBERTa/T5 모델, tokenizer, Trainer |
| `datasets` | 4.8.5 | GLUE 로딩과 Dataset 처리 |
| `evaluate` | 0.4.6 | 평가 패키지 의존성 |
| `peft` | 0.19.1 | LoRA 및 QLoRA adapter 구성 |
| `bitsandbytes` | 0.49.2 | Int8, FP4, NF4 양자화와 paged optimizer |
| `accelerate` | 1.13.0 | Transformers/PEFT의 장치 실행 지원 |
| `rouge-score` | 0.1.2 | SNI RougeL 계산 |
| `scikit-learn` | 1.7.2 | GLUE metric 계산 지원 |
| `PyYAML` | 5.4.1 | YAML config와 manifest 로딩 |
| `numpy` | 2.2.6 | metric 및 배열 처리 |

주요 전이 의존성 버전은 다음과 같습니다.

```text
huggingface-hub==0.36.2
tokenizers==0.22.2
safetensors==0.7.0
pyarrow==24.0.0
pandas==2.3.3
scipy==1.15.3
nvidia-cudnn-cu12==9.10.2.21
nvidia-nccl-cu12==2.27.5
```

현재 환경과 같은 주요 직접 의존성 버전을 설치하려면 다음 명령을 사용할 수 있습니다. PyTorch CUDA wheel 설치 방식은 대상 시스템의 CUDA와 PyTorch 배포 정책에 맞춰 별도로 확인해야 합니다.

```bash
pip install \
  torch==2.10.0 \
  transformers==4.57.6 \
  datasets==4.8.5 \
  evaluate==0.4.6 \
  peft==0.19.1 \
  bitsandbytes==0.49.2 \
  accelerate==1.13.0 \
  rouge-score==0.1.2 \
  scikit-learn==1.7.2 \
  PyYAML==5.4.1 \
  numpy==2.2.6
```

GLUE 실험은 `datasets`, `transformers`, `torch`, `scikit-learn`을 공통으로 사용합니다. LoRA/QLoRA 방법은 추가로 `peft`를 사용하고, 양자화된 QLoRA 방법은 `bitsandbytes`가 필요합니다. SNI 실험은 여기에 로컬 Natural Instructions 데이터와 `rouge-score`를 추가로 사용합니다.

## 첫 sanity check 실행

LoRA BF16:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/t5_lora_bf16.yaml
```

QLoRA NF4 + double quantization:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/t5_qlora_nf4.yaml
```

기본 sanity config는 `toy_sni`를 사용합니다. Hugging Face의 Super-NaturalInstructions 전체 파일을 내려받기 전에 학습/평가/CSV 로깅 경로를 빠르게 확인하기 위한 설정입니다. 이 결과는 코드 경로 확인용이며 논문 수치와 비교하면 안 됩니다.

## 결과 저장 위치

모든 실험 결과는 기본적으로 다음 CSV에 append됩니다.

```text
reproduce_table3/results/table3_runs.csv
```

각 행에는 timestamp, task group, dataset, model name, method, quantization 설정, LoRA 설정, 학습 설정, seed, metric, CUDA peak memory가 기록됩니다.

결과 해석 문서와 중간 보고서는 CSV와 분리해 다음 폴더에 둡니다.

```text
reproduce_table3/results/reports/
```

실험 로그는 실행 중인 프로세스와의 호환성을 위해 다음 폴더에 유지합니다.

```text
reproduce_table3/logs/
```

## 주요 결과 CSV의 실험 재실행

아래 명령은 모두 저장소 루트에서 실행합니다. 결과 CSV 자체는 실행 입력이 아니며, 각 CSV에 대응하는 YAML config와 manifest가 실제 실행 설정입니다.

러너는 기존 CSV를 덮어쓰지 않고 결과 행을 append합니다. 기존 결과와 새 결과를 분리하려면 재실행 전에 CSV를 백업하거나 각 config의 `results_csv`를 새 파일명으로 변경해야 합니다.

예:

```bash
cp reproduce_table3/results/table3_glue_table3.csv \
  reproduce_table3/results/table3_glue_table3.before_rerun.csv
```

### `table3_glue_bf16_reference_lr_sweep.csv`

대응 결과 파일:

```text
reproduce_table3/results/table3_glue_bf16_reference_lr_sweep.csv
```

표준 BF16 reference sweep은 RoBERTa-large full BF16 finetuning으로, accuracy 기반 GLUE 태스크 6개와 learning rate 3개를 조합한 18개 실험입니다.

- 태스크: SST-2, MRPC, QQP, MNLI, QNLI, RTE
- learning rate: `1e-5`, `2e-5`, `3e-5`
- epoch: 10
- seed: 42
- config: `reproduce_table3/configs/glue_bf16_reference/*.yaml`
- manifest: `reproduce_table3/configs/glue_bf16_reference_manifest.yaml`

config와 manifest를 다시 생성하려면 다음을 실행합니다.

```bash
python3 -m reproduce_table3.experiment_files.glue.make_glue_bf16_reference_configs
```

표준 18개 실험을 순차 실행합니다.

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --manifest reproduce_table3/configs/glue_bf16_reference_manifest.yaml \
  --continue-on-error
```

이 CSV에는 마감 시간 내 실행을 위해 추가한 빠른 보충 실험도 함께 기록될 수 있습니다. 보충 설정은 learning rate `2e-5`, 단일 GPU train batch size 32, gradient accumulation 1을 사용합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --manifest reproduce_table3/configs/glue_bf16_deadline_gpu0_manifest.yaml \
  --continue-on-error

CUDA_VISIBLE_DEVICES=1 python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --manifest reproduce_table3/configs/glue_bf16_deadline_gpu1_manifest.yaml \
  --continue-on-error
```

위 두 명령은 서로 다른 터미널에서 동시에 실행할 수 있습니다. GPU 0은 MNLI를 실행하고, GPU 1은 RTE와 QNLI를 순차 실행합니다. 표준 sweep과 보충 실험은 batch 설정이 다르므로 결과 비교 시 CSV의 `batch_size`와 `gradient_accumulation_steps`를 함께 확인해야 합니다.

### `table3_glue_table3.csv`

대응 결과 파일:

```text
reproduce_table3/results/table3_glue_table3.csv
```

이 manifest는 RoBERTa-large와 GLUE 8개 태스크에 다음 4개 방법을 적용한 32개 실험을 포함합니다.

- 태스크: CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE
- 방법: full BF16, LoRA BF16, QLoRA Int8, QLoRA FP4
- config: `reproduce_table3/configs/glue_table3/*.yaml`
- manifest: `reproduce_table3/configs/glue_table3_manifest.yaml`

config와 manifest를 다시 생성하려면 다음을 실행합니다.

```bash
python3 -m reproduce_table3.experiment_files.glue.make_glue_table3_configs
```

전체 32개 실험을 순차 실행합니다.

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --manifest reproduce_table3/configs/glue_table3_manifest.yaml \
  --continue-on-error
```

특정 실험 하나만 재실행하려면 manifest 대신 config를 지정합니다.

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/glue_table3/mnli_qlora_fp4.yaml
```

### `table3_sni_local_50k.csv`

대응 결과 파일:

```text
reproduce_table3/results/table3_sni_local_50k.csv
```

이 manifest는 AllenAI Natural Instructions 로컬 데이터에서 train 50,000개와 eval 1,000개를 사용하며, T5 크기 3개와 방법 2개를 조합한 6개 실험을 포함합니다.

- 모델: `google/t5-v1_1-small`, `google/t5-v1_1-base`, `google/t5-v1_1-large`
- 방법: LoRA BF16, QLoRA NF4 + double quantization
- epoch: 1
- seed: 42
- config: `reproduce_table3/configs/sni_local_50k/*.yaml`
- manifest: `reproduce_table3/configs/sni_local_50k_manifest.yaml`

먼저 로컬 Natural Instructions repo가 있어야 합니다.

```bash
git clone --depth 1 https://github.com/allenai/natural-instructions.git \
  reproduce_table3/data/natural-instructions
```

이미 디렉터리가 있다면 다시 clone할 필요가 없습니다. 데이터와 split을 확인합니다.

```bash
python3 -m reproduce_table3.experiment_files.sni.inspect_local_sni \
  --repo reproduce_table3/data/natural-instructions \
  --split default \
  --eval-instances-per-task 100
```

config와 manifest를 다시 생성하려면 다음을 실행합니다.

```bash
python3 -m reproduce_table3.experiment_files.sni.make_sni_local_50k_configs
```

전체 6개 실험을 순차 실행합니다.

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --manifest reproduce_table3/configs/sni_local_50k_manifest.yaml \
  --continue-on-error
```

QLoRA NF4 실험은 CUDA에서 동작하는 `bitsandbytes`가 필요합니다. GLUE 데이터 다운로드에 실패하면 현재 loader가 toy dataset으로 fallback할 수 있으므로, 실제 결과를 수집할 때는 로그에 `[dataset fallback]`이 없는지 확인해야 합니다.

## AllenAI 원본 Natural Instructions repo 사용

Hugging Face flatten dataset 대신 AllenAI 원본 JSON task와 공식 split 파일을 직접 쓰려면 먼저 repo를 내려받습니다.

```bash
git clone --depth 1 https://github.com/allenai/natural-instructions.git \
  reproduce_table3/data/natural-instructions
```

로컬 repo의 split과 instance 수를 확인합니다.

```bash
python3 -m reproduce_table3.experiment_files.sni.inspect_local_sni \
  --repo reproduce_table3/data/natural-instructions \
  --split default \
  --eval-instances-per-task 100
```

로컬 SNI repo pilot 실행:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/sni_local_t5_780m_qlora_nf4_dq.yaml
```

전체 train split 1 epoch 실행 템플릿:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/sni_local_t5_780m_qlora_nf4_dq_full.yaml
```

주의: 원본 repo의 `splits/default/test_tasks.txt`는 평가 task 목록이고, Natural Instructions README는 각 test task의 `Instances[:100]`을 평가에 쓰는 방식을 설명합니다. 이 프레임워크의 로컬 loader도 기본적으로 `local_sni_eval_instances_per_task: 100`을 사용합니다. `max_eval_samples: 1000`을 지정하면 그중 앞 1000개만 평가합니다.

## 다른 설정 실행

T5 FP4:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/t5_qlora_fp4.yaml
```

T5 Int8:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/t5_qlora_int8.yaml
```

RoBERTa GLUE LoRA BF16:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/roberta_lora_bf16.yaml
```

RoBERTa GLUE QLoRA FP4:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --config reproduce_table3/configs/roberta_qlora_fp4.yaml
```

RoBERTa-large로 확장하려면 `configs/roberta_lora_bf16.yaml` 또는 `configs/roberta_qlora_fp4.yaml`의 `model_name_or_path`를 `FacebookAI/roberta-large`로 바꿉니다.

## 전체 Table 3 config 생성 및 실행

원 Table 3의 실험 축을 모두 config로 생성하려면 다음을 실행합니다.

```bash
python3 -m reproduce_table3.experiment_files.config_generators.make_table3_configs
```

생성물:

- `reproduce_table3/configs/table3_full/*.yaml`
- `reproduce_table3/configs/table3_full_manifest.yaml`

전체 manifest 실행:

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --manifest reproduce_table3/configs/table3_full_manifest.yaml \
  --continue-on-error
```

주의: 이 명령은 T5-3B/11B, RoBERTa-large, full SNI 다운로드와 학습을 포함하므로 GPU 메모리와 시간이 많이 필요합니다. RTX 3090 24GB 2장만으로는 T5-11B BF16 full finetuning 같은 항목은 현실적으로 실행하기 어렵습니다.

## 더 큰 T5 모델로 확장

T5-base 이상으로 확장하려면 T5 config의 `model_name_or_path`를 바꿉니다.

예:

```yaml
model_name_or_path: t5-base
```

원 Table 3에 가까운 실험으로 가려면 다음을 별도로 맞춰야 합니다.

- 원 논문과 동일한 Super-NaturalInstructions split과 전처리
- 원 논문의 학습 step, batch size, learning rate, warmup, scheduler
- T5-80M, 250M, 780M, 3B, 11B에 해당하는 정확한 checkpoint
- BF16 full finetuning과 BF16 replication baseline
- 여러 seed 반복 및 평균/분산 보고
- 충분한 GPU 메모리와 학습 시간

## 폴더 구조

- `experiment_files/glue/`: RoBERTa GLUE 학습, 평가, GLUE config 생성 로직
- `experiment_files/sni/`: T5 Super-NaturalInstructions 학습, RougeL 평가, SNI config 생성 로직
- `experiment_files/common/`: GLUE와 SNI가 함께 쓰는 공통 유틸리티
- `experiment_files/runner/`: config 또는 manifest 기반 실험 실행 진입점
- `experiment_files/config_generators/`: Table 3/hardware-scope manifest 생성 로직
- `configs/`: YAML 실험 설정과 manifest
- `results/`: CSV 결과 파일
- `results/reports/`: 실행 상태, 실험 요약, 논문 비교 Markdown 문서
- `logs/`: 실험 실행 로그
- `outputs/`: Trainer 출력 디렉터리
- `data/`: 로컬 데이터셋 저장 위치

## 구현된 파일

- `experiment_files/sni/train_t5_sni.py`: T5 + Super-NaturalInstructions 또는 toy instruction 데이터 학습, RougeL 평가, CSV 저장
- `experiment_files/sni/eval_t5_rougel.py`: RougeL 계산 유틸리티와 JSONL 평가 CLI
- `experiment_files/glue/train_roberta_glue.py`: RoBERTa + GLUE 학습, Accuracy 평가, CSV 저장
- `experiment_files/glue/eval_glue.py`: Accuracy 계산 유틸리티와 JSONL 평가 CLI
- `experiment_files/runner/run_experiments.py`: YAML 설정의 `task_group`에 따라 T5 또는 GLUE 학습 스크립트 dispatch
- `experiment_files/config_generators/make_table3_configs.py`: 원 Table 3 전체 실험 축에 대한 YAML config와 manifest 생성
- `experiment_files/common/table3_common.py`: YAML 로딩, seed 설정, bitsandbytes 설정, CSV 로깅 공통 코드
- `configs/*.yaml`: 작은 규모 sanity check용 기본 설정
- `configs/table3_full/*.yaml`: full Table 3 실행용 설정
- `results/reports/EXECUTION_STATUS.md`: 현재까지 실제 실행한 실험과 실패/중단 사유 기록

## 알려진 한계

이 코드는 Table 3의 전체 재현 결과가 아닙니다. 현재 목적은 작은 모델과 작은 샘플 수에서 학습, 평가, 로깅 경로가 정상 동작하는지 확인하는 것입니다.

또한 Super-NaturalInstructions 데이터셋의 Hugging Face schema가 환경마다 다를 수 있어 입력/출력 필드를 최대한 유연하게 읽도록 구현했습니다. 정확한 논문 재현 단계에서는 원 논문 코드와 동일한 데이터 처리 방식을 별도로 고정해야 합니다.
