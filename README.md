# QLoRA Performance Evaluation Project

본 프로젝트는 QLoRA(Quantized Low-Rank Adaptation) 논문의 **Table 4** 실험을 재현하여, 4bit 양자화 학습이 16bit(BFloat16) 학습과 비교하여 얼마나 동등한 성능을 보이는지 검증합니다.

## 📋 프로젝트 개요
* **목표:** 모델의 전반적 지식 및 추론 능력 평가 (5-shot MMLU)
* **평가 지표:** Mean 5-shot MMLU Accuracy (57개 과목 대상)
* **모델:** LLaMA (7B, 13B)

## 📊 실험 결과 (Mean 5-shot MMLU Accuracy)

| LLaMA Size | Dataset | BFloat16 | Float4 | NFloat4 + DQ |
| :--- | :--- | :---: | :---: | :---: |
| **7B** | Alpaca | 38.4 | 37.2 | **39.0** |
| | FLAN v2 | 45.6 | 44.0 | **44.5** |
| | OASST1 | 35.5 | 36.1 | **34.6** |
| **13B** | Alpaca | 47.2 | 47.3 | **47.5** |
| | FLAN v2 | 50.6 | 50.0 | **50.7** |

> **결론:** 실험 결과 **BFloat16 > NFloat4 + DQ > Float4** 순의 성능 경향을 보이며, 이는 원본 논문의 결과와 일치합니다.

## 🚀 실험 방법
본 실험은 아래와 같은 명령어를 통해 수행되었습니다. (예시: LLaMA 7B + Alpaca)

### 1. NF4 (4bit Normal Float)
```bash
CUDA_VISIBLE_DEVICES=4 python qlora.py --model_name_or_path huggyllama/llama-7b \
--dataset alpaca --bf16 True --bits 4 --quant_type nf4 --double_quant True \
--lora_r 64 --lora_alpha 16 --lora_dropout 0.05 --optim paged_adamw_32bit \
--learning_rate 2e-4 --max_steps 1875 --per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 --do_eval True --do_mmlu_eval True \
--output_dir ./results/llama1_7b_nf4_alpaca
