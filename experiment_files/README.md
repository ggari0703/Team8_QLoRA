# 실험 실행 파일 구조

이 폴더에는 실제 학습, 평가, config 생성 로직이 들어 있다.

- `glue/`: RoBERTa GLUE 학습, 평가, GLUE config 생성 코드이다.
- `sni/`: T5 Super-NaturalInstructions 학습, RougeL 평가, SNI config 생성 코드이다.
- `common/`: 두 실험이 함께 쓰는 YAML 로딩, seed, quantization, CSV 기록 코드이다.
- `runner/`: YAML config 또는 manifest를 읽어 GLUE/SNI 실험으로 dispatch하는 실행 진입점이다.
- `config_generators/`: GLUE와 SNI를 함께 포함하는 Table 3/hardware-scope manifest 생성 코드이다.

루트에는 Python wrapper를 두지 않는다. GitHub 메인 화면에는 폴더와 `README.md`, `requirements.txt`만 보이도록 구성한다.

현재 작업 디렉터리(`/home/gyuseong`) 기준 실행 예시는 다음과 같다.

```bash
python3 -m reproduce_table3.experiment_files.runner.run_experiments \
  --manifest reproduce_table3/configs/glue_table3_manifest.yaml
```
