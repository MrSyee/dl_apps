# 쓸모있는 AI 서비스 만들기
"쓸모있는 AI 서비스 만들기"에서 소개하는 예제들을 정리한 레포입니다.

## Contents
### OCR: [Handwritting Recognition](ocr/)
![handwritten_ocr](https://github.com/MrSyee/dl_apps/assets/17582508/24998a51-bc4c-4b1f-b20b-34722dbab708)

### Segmentation: [Remove background](segmentation/)
![remove_background_sam](https://github.com/MrSyee/dl_apps/assets/17582508/b6447983-7797-4b9d-bc8c-e219600435cb)

### NLP: [News Article Search Chatbot](nlp/)
![news_article_chatbot](https://github.com/MrSyee/dl_apps/assets/17582508/11993e84-f296-4759-b709-913bc601b640)

### Speech Recognition: [YouTube Subtitle Generator](speech_recognition)
![youtube_subtitle_generator](https://github.com/MrSyee/dl_apps/assets/17582508/4e628115-e7eb-496a-96c7-4d54c0562084)

### Image Generation: [Sketch to Image Generator](image_generation)
![sketch_to_image](https://github.com/MrSyee/dl_apps/assets/17582508/9f1e60c3-f612-499c-90ed-e42d2a6ad379)


## Patch Notes
- 2024.07.17: Speech Recognition 예제의 pytube 이슈 해결을 위한 셀 추가
  - Speech Recognition 예제에서 pytube 15.0.0 버전에서 변경된 youtube URL을 제대로 처리하지 못하고 에러가 발생.
  - [이슈](https://github.com/pytube/pytube/issues/1954)에서 해결방안이 있으나 공식적으로 pytube 라이브러리에 업데이트 되지 않음.
  - 문제를 해결된 버전의 pytube를 사용하기 위해 임시로 패키지 설치 셀을 추가함.
    ```
    !pip install -U git+https://github.com/an-dutra/pytube.git@patch-1
    ```
