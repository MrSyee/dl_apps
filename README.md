# 쓸모있는 AI 서비스 만들기
"쓸모있는 AI 서비스 만들기"에서 소개하는 예제들을 정리한 레포입니다.

<img src="https://github.com/user-attachments/assets/4e906e73-8bf2-49ba-9af4-348d05385dec" alt="drawing" width="200"/>

- 교보: https://product.kyobobook.co.kr/detail/S000213842209
- 예스: https://www.yes24.com/Product/Goods/129119345
- 알라딘: https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=343293888

## Patch Notes
> ⚠️ 라이브러리 버전의 변화, 업데이트 등의 이유로 발생하는 오류로 인해 책 내용과 다르게 수정된 내용이 있습니다. 주기적으로 업데이트할 예정이니 참고바랍니다. 실습 중 문제가 있는 예제가 있다면 [이슈](https://github.com/MrSyee/dl_apps/issues)에 남겨주세요.

- 2024.10.01: Speech Recognition 예제의 pytube 이슈 해결을 위한 라이브러리 변경
  - youtube 관련해 버전 패치를 해도 문제가 해결되지 않음.
  - 위 문제가 해결된 새로운 라이브러리인 pytubefix로 라이브러리 변경
- 2024.07.17: Speech Recognition 예제의 pytube 이슈 해결을 위한 셀 추가
  - Speech Recognition 예제에서 pytube 15.0.0 버전에서 변경된 youtube URL을 제대로 처리하지 못하고 에러가 발생.
  - [이슈](https://github.com/pytube/pytube/issues/1954)에서 해결방안이 있으나 공식적으로 pytube 라이브러리에 업데이트 되지 않음.
  - 문제를 해결된 버전의 pytube를 사용하기 위해 임시로 패키지 설치 셀을 추가함.
    ```
    !pip install -U git+https://github.com/an-dutra/pytube.git@patch-1
    ```

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
