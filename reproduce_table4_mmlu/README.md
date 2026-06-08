# QLoRA Performance Evaluation Project
## ref: https://github.com/artidoro/qlora

본 프로젝트는 QLoRA(Quantized Low-Rank Adaptation) 논문의 **Table 4** 실험을 재현하여, 4bit 양자화 학습이 16bit(BFloat16) 학습과 비교하여 얼마나 동등한 성능을 보이는지 검증합니다.

## 프로젝트 개요
* **목표:** 모델의 전반적 지식 및 추론 능력 평가 (5-shot MMLU)
* **평가 지표:** Mean 5-shot MMLU Accuracy (57개 과목 대상)
* **모델:** LLaMA (7B, 13B)
* **데이터셋:** Alpaca dataset, FLAN v2 dataset, OASST dataset(7B)
  * Alpaca dataset - 52,002개 GPT-3.5로 생성한 instruction 데이터. 다양한 태스크 포함
  *  FLAN v2 dataset - 구글이 만든 대규모 instruction 데이터. 지식/추론 중심, 서브샘플링 한 450,000개만 사용  

## 실험 내용 및 결과 (Mean 5-shot MMLU Accuracy)
* 57개 과목 (수학, 역사, 법학, 의학, 물리 등) 4지선다 문제로 LLM의 지식과 추론 능력을 평가하는 벤치마크
* 점수가 높을수록 모델의 전반적 지식 수준이 높음
* 문제 풀기 전 같은 과목 예시 문제 5개 제공 후 평가

| LLaMA Size | Dataset | BFloat16(ours) | Float4(ours) | NFloat4 + DQ(ours) |
| :--- | :--- | :---: | :---: | :---: |
| **7B** | Alpaca | 38.4(35.3) | 37.2(36.9) | **39.0**(36.4) | 
| | FLAN v2 | 45.6(45.7) | 44.0(35.2) | 44.5(36.0) |
| **13B** | Alpaca | 47.2(46.5) | 47.3(46.2) | 47.5(46.1) |
| | FLAN v2 | 50.6(41.3) | 50.0(40.9) | 50.7(43.2) |
| | **MEAN** | 45.5(42.2) | 44.6(39.8) | 45.4(40.4) |

> **결론:**
* 실험 결과 **BFloat16 > NFloat4 + DQ > Float4** 순의 성능 경향을 보이며, 이는 원본 논문의 결과와 일치합니다.
* 특히, 4bit 양자화(NF4/FP4)를 적용하더라도 16bit 모델 대비 성능 저하가 미미하여 매우 효율적인 학습이 가능함을 확인하였습니다.

##  실험 방법
* 본 실험은 아래와 같은 명령어를 통해 수행되었습니다. (예시: LLaMA 7B + Alpaca)
* claude를 코드 작성 및 에러 디버깅에 사용하였습니다.

### 0. enviroment settings
      pip install -U -r requirements.txt

### 1. NFloat4 + DQ
    CUDA_VISIBLE_DEVICES=4 python qlora.py --model_name_or_path huggyllama/llama-7b \
    --dataset alpaca --bf16 True --bits 4 --quant_type nf4 --double_quant True \
    --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 --optim paged_adamw_32bit \
    --learning_rate 2e-4 --max_steps 1875 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 --do_eval True --do_mmlu_eval True \
    --output_dir ./results/llama1_7b_nf4_alpaca

### 2. Float4
    CUDA_VISIBLE_DEVICES=5 python qlora.py --model_name_or_path huggyllama/llama-7b \
    --dataset alpaca --bf16 True --bits 4 --quant_type nf4 --double_quant True \
    --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 --optim paged_adamw_32bit \
    --learning_rate 2e-4 --max_steps 1875 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 --do_eval True --do_mmlu_eval True \
    --output_dir ./results/llama1_7b_nf4_alpaca

### 3. BFloat16
    CUDA_VISIBLE_DEVICES=6,7 python qlora.py --model_name_or_path huggyllama/llama-7b \
    --dataset alpaca --bf16 True --bits 16 --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 --optim paged_adamw_32bit \
    --learning_rate 2e-4 --max_steps 1875 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --max_memory_MB 20000 --gradient_checkpointing False --do_eval True --do_mmlu_eval True --mmlu_dataset mmlu-fs \
    --mmlu_split test --output_dir ./results/llama1_7b_bf16_alpaca
