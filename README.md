# 구글 BERT의 정석

<a href="https://www.packtpub.com/product/getting-started-with-google-bert/9781838821593?utm_source=github&utm_medium=repository&utm_campaign=9781838821593"><img src="https://static.packt-cdn.com/products/9781838821593/cover/smaller" alt="Getting Started with Google BERT" height="256px" align="right"></a>

이곳은 한빛미디어에서 번역 및 출판한 [구글 BERT의 정석(원문: Getting Started with Google BERT, Packt)](https://www.packtpub.com/product/getting-started-with-google-bert/9781838821593?utm_source=github&utm_medium=repository&utm_campaign=9781838821593)의 소스 코드 저장소입니다.

**BERT를 활용한 최신 자연어 처리 모델을 만들어보고 학습시켜보세요!**

## 이 책은 주제는 무엇인가요?
BERT(트랜스포머 기반 양방향 인코더)는 괄목할만한 결과를 보여주며 자연어 처리 필드에 혁신을 가져왔습니다. 이 책은 구글 BERT 아키텍처에 대한 이해를 도와주는 입문서입니다. 트랜스포머 아키텍처의 상세한 설명과 더불어 독자로 하여금 트랜스포머의 인코더와 디코더의 작동원리를 이해하는데 도움을 줄 것입니다.

허깅페이스 트랜스포머 라이브러리를 활용하여 BERT 모델이 어떻게 사전 학습되고 감적분석과 텍스트 요약 같은 다운스트림 작업을 위해 사전 학습된 모델을 재학습(fine-tuning)하여 어떻게 활용할 수 있는지를 배우며 BERT 아키텍처에 대해 알아볼 것입니다. 더 나아가, ALBERT, RoBERTa, 그리고 ELECTRA와 같은 BERT의 다양한 종류에 대해 배우고, 질의 응답 과제에 사용되는 SpanBERT에 대해 살펴봅니다. 또한, 지식 증류에 기반하고 기존 BERT 모델보다 가볍고 빠른 BERT의 아종인 DistillBERT와 SapnBERT에 대해서도 다룰 것입니다. 이 책은 MBERT, XLM, 그리고 XLM-R에 대해 상세히 살펴보고, 문장의 표현 얻는데 사용되는 sentence-BERT에 대해 소개합니다. 마지막으로, BioBERT와 ClinicalBERT 같은 도메인 특화 모델과 VideoBERT라 불리는 흥미로운 모델에 대해서도 다룹니다.

책을 끝까지 읽고나면, 실용적인 자연어 처리 작업을 수행하는데에 BERT와 BERT의 아종을 숙련되게 사용할 수 있을 것입니다.

이 책은 다음의 흥미로운 내용을 다룹니다:
* 트랜스포머 모델의 기초에 대한 이해
* BERT의 작용 매커니즘과 두 사전 작업인 마스크 된 언어 모델(Masked Language Model;MLM)과 다음 문장 예측(Next Sentence Prediction;NSP)으로 사전 학습을 시키는 방법
* 다운스트림 작업에 BERT를 재학습
* ALBERT, RoBERTa, ELECTRA, 그리고 SpanBERT 모델에 대한 이해
* 지식 증류 기반 모델들에 대한 이해
* XLM과 XLM-R과 같은 corss-lingual 모델에 대한 이해
* Sentence-BERt, VideoBERT, 그리고 BART 모델에 대한 이해

이 책이 당신에게 딱 맞는 책이라면, 당장 [구입](https://www.amazon.com/dp/1838821597)하세요!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## 알아둘 것
모든 코드는 폴더 단위로 구성되었습니다.

코드 맛보기:
```
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(train_df = df,
                                                                   text_column = 'reviewText',
                                                                   label_columns=['sentiment'],
                                                                   maxlen=100,
                                                                   max_features=100000,
                                                                   preprocess_mode='bert',
                                                                   val_pct=0.1)

```

****
이 책은 BERT를 활용하여 자연어 이해와 같은 자연어 처리 작업을 효율적으로 만드는 방법을 찾고 있는 자연어 처리 전문과와 데이터 사인어티스트를 위한 것입니다. 자연어 처리의 기본적인 개념과 딥러닝에 대한 기초를 전제로 하며, 이 책에서는 다루지 않습니다. 책의 내용을 최대한 숙지하려면, 구글 Colab을 활용하여 이 책에서 제공 된 모든 코드를 직접 실행해보세요.

아래의 소프트웨어와 하드웨어 필요 사양을 갖추면, 제공 된 모든 코드를 수월하게 실행할 수 있습니다 (챕터 1,2,5는 코드 파일을 포함하지 않습니다).

### 소프웨어 하드웨어 필요 사양

| 챕터  | 소프트웨어 사양                                                                                   | 운영체제 사양                        |
| -------- | ----------------------------------------------------------------------------------------------------| -----------------------------------|
|    3     |   Google Colab / Python 3.x                                                                         | Windows, Mac OS X, and Linux (Any) |
|    4     |   Google Colab / Python 3.x                                                                         | Windows, Mac OS X, and Linux (Any) |
|    6     |   Google Colab / Python 3.x                                                                         | Windows, Mac OS X, and Linux (Any) |
|    7     |   Google Colab / Python 3.x                                                                         | Windows, Mac OS X, and Linux (Any) |
|    8     |   Google Colab / Python 3.x                                                                         | Windows, Mac OS X, and Linux (Any) |
|    9     |   Google Colab / Python 3.x                                                                         | Windows, Mac OS X, and Linux (Any) |


책에 있는 흑백 이미지의 컬러 버젼을 제공합니다. [다운로드](https://static.packt-cdn.com/downloads/9781838821593_ColorImages.pdf).

## 오탈자

* 26쪽 (챕터 1, positional encoding으로 position에 대해 배우기)

수정 후:

<img src = "https://github.com/PacktPublishing/Getting-Started-with-Google-BERT/blob/main/Errata_images/chapter1_10000.PNG">

수정 전: 

<img src = "https://github.com/PacktPublishing/Getting-Started-with-Google-BERT/blob/main/Errata_images/chapter1_1000.png">

### 관련 도서 <독자들이 흥미를 느낄 다른 책들>
* Hands-On Python Natural Language Processing [[Packt]](https://www.packtpub.com/product/hands-on-python-natural-language-processing/9781838989590) [[Amazon]](https://www.amazon.com/dp/1838989595)

* The Applied AI and Natural Language Processing Workshop [[Packt]](https://www.packtpub.com/product/the-applied-ai-and-natural-language-processing-workshop/9781800208742) [[Amazon]](https://www.amazon.com/dp/B08Q8GNTGT)

### 저자 소개
**Sudharsan Ravichandiran**은 데이터 사이언티스트, 연구원, 그리고 베스트 셀러 작가이다. Anna University에서 Information Technology로 학사 학위를 마쳤다. 그의 관심 연구 분야는 자연어 처리와 컴퓨터 비전을 포함하여 딥러닝과 강화학습의 실용적인 구현이다. 저자는 오픈 소스에 기여하고 StackOverflow의 질문에 답하는 것을 즐긴다. 또한, 베스트 셀러 Hands-On Reinforcement Learning with Pyothon (Packt 출판) 의 저자이기도 하다.

### 저장소 번역 계기와 주저리
Deep learning 연구와 개발을 하며, 최근 자연어 처리에 관심을 갖게 되었는데 BERT가 급속도로 발전한 반면 BERT에 대한 백과사전식 review/survery paper나 책은 찾기가 힘들었습니다. 이 책은 BERT 입문서로서 그에 대한 갈증을 해결하기에 충분하다고 느끼게 되면서 영어로 된 Github repository를 번역하게 되었습니다. 연구향으로 접근한 책은 아니지만, 실용적인 내용을 위주로 많이 다뤘다고 생각하여 입문 단계에서 좋은 참고서라고 생각합니다 (신기하게도 쉽게 잘 풀었더군요). 번역도 처음이고 자연어 처리 쪽 전공자도 아니어서 잘못 된 용어(번역)를 사용할 수도 있는데 감안해주시면 감사하겠습니다. 혹시 해당 repository가 저작권 문제가 있을 경우, 바로 조치하도록 하겠습니다 (chaeunl7765@gmail.com).

일을 하면서 한 챕터 한 챕터 번역하는 것이라 시간이 다소 소요될 수도 있고, 코드를 돌려보면서 "이런 코드는 더 추가되면 좋겠는데?" 라고 생각들면 추가한 코드는 표시해두겠습니다. 우리 모두 즐겁게 개발 합시다!
