# Wav2Vec2

## Abstract

<details>
<summary>en</summary>

We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech recognition with limited amounts of labeled data.1
</details>

음성 오디오만으로 강력한 표현을 학습한 후 전사된 음성을 미세 조정하면 개념적으로 더 간단하면서도 최고의 반지도 방식보다 성능이 뛰어날 수 있음을 처음으로 보여줍니다. wav2vec 2.0은 잠재 공간에서 음성 입력을 마스킹하고 공동으로 학습된 잠재 표현의 양자화를 통해 정의된 대조 작업을 해결합니다. 라이브러스피치의 모든 라벨링된 데이터를 사용한 실험은 클린/기타 테스트 세트에서 1.8/3.3의 WER을 달성했습니다. 라벨링된 데이터의 양을 1시간으로 줄였을 때, 100시간 하위 세트에서는 100배 적은 라벨링된 데이터를 사용하면서도 wav2vec 2.0이 이전 기술보다 더 나은 성능을 보였습니다. 단 10분의 라벨링된 데이터와 53,000시간의 라벨링되지 않은 데이터에 대한 사전 학습을 사용해도 4.8/8.2의 WER을 달성할 수 있습니다. 이는 제한된 양의 라벨링된 데이터로도 음성 인식이 가능하다는 것을 보여줍니다.1

## Introduction

<details>
<summary>en</summary>

Neural networks benefit from large quantities of labeled training data. However, in many settings labeled data is much harder to come by than unlabeled data: current speech recognition systems require thousands of hours of transcribed speech to reach acceptable performance which is not available for the vast majority of the nearly 7,000 languages spoken worldwide [31]. Learning purely from labeled examples does not resemble language acquisition in humans: infants learn language by listening to adults around them - a process that requires learning good representations of speech.

In machine learning, self-supervised learning has emerged as a paradigm to learn general data representations from unlabeled examples and to fine-tune the model on labeled data. This has been particularly successful for natural language processing [43, 45, 9] and is an active research area for computer vision [20, 2, 36, 19, 6]

In this paper, we present a framework for self-supervised learning of representations from raw audio data. Our approach encodes speech audio via a multi-layer convolutional neural network and then masks spans of the resulting latent speech representations [26, 56], similar to masked language modeling [9]. The latent representations are fed to a Transformer network to build contextualized representations and the model is trained via a contrastive task where the true latent is to be distinguished from distractors [54, 49, 48, 28] (§ 2)

As part of training, we learn discrete speech units [53, 32, 7, 18] via a gumbel softmax [24, 5] to represent the latent representations in the contrastive task (Figure 1) which we find to be more effective than non-quantized targets. After pre-training on unlabeled speech, the model is fine-tuned on labeled data with a Connectionist Temporal Classification (CTC) loss [14, 4] to be used for downstream speech recognition tasks (§ 3)

Previous work learned a quantization of the data followed by a contextualized representations with a self-attention model [5, 4], whereas our approach solves both problems end-to-end. Masking parts of the input with Transformer networks for speech has been explored [4, 26], but prior work relies either on a two-step pipeline or their model is trained by reconstructing the filter bank input features. Other related work includes learning representations from auto-encoding the input data [52, 11] or directly predicting future timesteps [8].

Our results show that jointly learning discrete speech units with contextualized representations achieves substantially better results than fixed units learned in a prior step [4]. We also demonstrate the feasibility of ultra-low resource speech recognition: when using only 10 minutes of labeled data, our approach achieves word error rate (WER) 4.8/8.2 on the clean/other test sets of Librispeech. We set a new state of the art on TIMIT phoneme recognition as well as the 100 hour clean subset of Librispeech. Moreover, when we lower the amount of labeled data to just one hour, we still outperform the previous state of the art self-training method of [42] while using 100 times less labeled data and the same amount of unlabeled data. When we use all 960 hours of labeled data from Librispeech, then our model achieves 1.8/3.3 WER (§ 4, § 5).
</details>

신경망은 대량의 라벨링된 훈련 데이터를 활용하면 큰 이점을 얻을 수 있습니다. 그러나 많은 환경에서 레이블이 지정된 데이터는 레이블이 지정되지 않은 데이터보다 훨씬 더 구하기 어렵습니다. 현재 음성 인식 시스템은 허용 가능한 성능에 도달하려면 수천 시간의 전사된 음성이 필요하며, 이는 전 세계에서 사용되는 약 7,000개의 언어 중 대다수에서 사용할 수 없는 수준입니다[31]. 라벨이 붙은 예시만 보고 학습하는 것은 인간의 언어 습득과 유사하지 않습니다. 유아는 주변 어른들의 말을 들으며 언어를 배우는데, 이 과정에서 좋은 음성 표현을 배워야 합니다.

머신러닝에서 자가 지도 학습은 레이블이 없는 예시로부터 일반적인 데이터 표현을 학습하고 레이블이 있는 데이터에 대해 모델을 미세 조정하는 패러다임으로 부상했습니다. 이는 자연어 처리에서 특히 성공적이었으며[43, 45, 9], 컴퓨터 비전에서 활발히 연구되고 있는 분야입니다[20, 2, 36, 19, 6].

이 백서에서는 원시 오디오 데이터의 표현에 대한 자가 지도 학습 프레임워크를 제시합니다. 우리의 접근 방식은 다층 컨볼루션 신경망을 통해 음성 오디오를 인코딩한 다음, 마스크 언어 모델링[9]과 유사하게 결과물인 잠재 음성 표현의 스팬을 마스킹합니다[26, 56]. 잠복 표현은 Transformer 네트워크에 공급되어 문맥화된 표현을 구축하고 모델은 실제 잠복과 방해 요소를 구별하는 대조 작업을 통해 훈련됩니다[54, 49, 48, 28](§ 2).

훈련의 일환으로 굼벨 소프트맥스[24, 5]를 통해 개별 음성 단위[53, 32, 7, 18]를 학습하여 대조 과제(그림 1)에서 잠재 표현을 표현하는데, 이는 정량화되지 않은 타깃보다 더 효과적이라고 판단했습니다. 라벨이 없는 음성에 대한 사전 학습 후, 모델은 연결주의 시간 분류(CTC) 손실[14, 4]이 있는 라벨이 있는 데이터에 대해 미세 조정되어 다운스트림 음성 인식 작업에 사용됩니다(§ 3).

이전 연구에서는 데이터의 양자화를 학습한 후 자기 주의 모델을 통해 문맥화된 표현을 학습했지만[5, 4], 우리의 접근 방식은 두 가지 문제를 엔드 투 엔드로 해결합니다. 음성용 트랜스포머 네트워크로 입력의 일부를 마스킹하는 방법이 연구되었지만[4, 26], 이전 연구에서는 2단계 파이프라인에 의존하거나 필터 뱅크 입력 특징을 재구성하여 모델을 학습시켰습니다. 다른 관련 작업으로는 입력 데이터를 자동 인코딩하여 표현을 학습하거나[52, 11] 미래의 시간 단계를 직접 예측하는 방법[8]이 있습니다.

연구 결과에 따르면 문맥화된 표현과 함께 개별 음성 단위를 공동으로 학습하면 이전 단계에서 학습한 고정 단위보다 훨씬 더 나은 결과를 얻을 수 있습니다[4]. 또한 초저자원 음성 인식의 실현 가능성도 입증했습니다. 10분 분량의 라벨링된 데이터만 사용할 경우, 우리의 접근 방식은 Librispeech의 클린/기타 테스트 세트에서 단어 오류율(WER) 4.8/8.2를 달성했습니다. 저희는 TIMIT 음소 인식과 Librispeech의 100시간 클린 하위 집합에 대한 새로운 최첨단 기술을 설정했습니다. 또한, 라벨링된 데이터의 양을 1시간으로 줄였을 때에도 100배 적은 라벨링된 데이터와 동일한 양의 라벨링되지 않은 데이터를 사용하면서도 이전의 최첨단 자가 학습 방법인 [42]보다 더 나은 성능을 보였습니다. 라이브러스피치에서 960시간의 라벨링된 데이터를 모두 사용할 경우, 우리 모델은 1.8/3.3의 WER을 달성합니다(§ 4, § 5).

## Model

<details>
<summary>en</summary>

Our model is composed of a multi-layer convolutional feature encoder f : X 7→ Z which takes as input raw audio X and outputs latent speech representations z1, . . . , zT for T time-steps. They are then fed to a Transformer g : Z 7→ C to build representations c1, . . . , cT capturing information from the entire sequence [9, 5, 4]. The output of the feature encoder is discretized to qt with a quantization module Z 7→ Q to represent the targets (Figure 1) in the self-supervised objective (§ 3.2). Compared to vq-wav2vec [5], our model builds context representations over continuous speech representations and self-attention captures dependencies over the entire sequence of latent representations end-to-end.

Feature encoder. The encoder consists of several blocks containing a temporal convolution followed by layer normalization [1] and a GELU activation function [21]. The raw waveform input to the encoder is normalized to zero mean and unit variance. The total stride of the encoder determines the number of time-steps T which are input to the Transformer (§ 4.2).

Contextualized representations with Transformers. The output of the feature encoder is fed to a context network which follows the Transformer architecture [55, 9, 33]. Instead of fixed positional embeddings which encode absolute positional information, we use a convolutional layer similar to [37, 4, 57] which acts as relative positional embedding. We add the output of the convolution followed by a GELU to the inputs and then apply layer normalization.

Quantization module. For self-supervised training we discretize the output of the feature encoder z to a finite set of speech representations via product quantization [25]. This choice led to good results in prior work which learned discrete units in a first step followed by learning contextualized representations [5]. Product quantization amounts to choosing quantized representations from multiple codebooks and concatenating them. Given G codebooks, or groups, with V entries e ∈ R V ×d/G, we choose one entry from each codebook and concatenate the resulting vectors e1, . . . , eG and apply a linear transformation R d 7→ R f to obtain q ∈ R f .

The Gumbel softmax enables choosing discrete codebook entries in a fully differentiable way [16, 24, 35]. We use the straight-through estimator [26] and setup G hard Gumbel softmax operations [24]. The feature encoder output z is mapped to l ∈ R G×V logits and the probabilities for choosing the v-th codebook entry for group g are

pg,v = exp(lg,v + nv)/τ PV k=1 exp(lg,k + nk)/τ

where τ is a non-negative temperature, n = − log(− log(u)) and u are uniform samples from U(0, 1). During the forward pass, codeword i is chosen by i = argmaxj pg,j and in the backward pass, the true gradient of the Gumbel softmax outputs is used.
</details>

이 모델은 다층 컨볼루션 특징 인코더 f : X 7→ Z로 구성되며, 원시 오디오 X를 입력으로 받아 잠재 음성 표현 z1, . . . , zT를 T 시간 단계에 대해 출력합니다. 그런 다음 트랜스포머 g : Z 7→ C로 공급되어 표현 c1, . . . 전체 시퀀스 [9, 5, 4]에서 정보를 캡처하는 cT를 구축합니다. 특징 인코더의 출력은 자체 감독 목표(§ 3.2)에서 목표를 나타내기 위해 양자화 모듈 Z 7→ Q를 사용하여 qt로 이산화됩니다(그림 1). vq-wav2vec [5]과 비교했을 때, 우리의 모델은 연속적인 음성 표현 위에 컨텍스트 표현을 구축하고 자기 주의는 잠재 표현의 전체 시퀀스에 대한 종속성을 엔드 투 엔드로 캡처합니다.

특징 인코더. 인코더는 시간 컨볼루션과 레이어 정규화[1] 및 GELU 활성화 함수[21]가 포함된 여러 블록으로 구성됩니다. 인코더에 입력되는 원시 파형은 평균과 단위 분산이 0으로 정규화됩니다. 인코더의 총 보폭은 트랜스포머에 입력되는 시간 스텝 T의 수를 결정합니다(§ 4.2).

트랜스포머를 사용한 컨텍스트화된 표현. 특징 인코더의 출력은 트랜스포머 아키텍처를 따르는 컨텍스트 네트워크에 공급됩니다 [55, 9, 33]. 절대 위치 정보를 인코딩하는 고정 위치 임베딩 대신 상대 위치 임베딩으로 작동하는 [37, 4, 57]과 유사한 컨볼루션 계층을 사용합니다. 컨볼루션의 출력과 GELU를 입력에 추가한 다음 레이어 정규화를 적용합니다.

양자화 모듈. 자가 지도 학습을 위해 특징 인코더 z의 출력을 제품 양자화를 통해 유한한 음성 표현 집합으로 이산화합니다[25]. 이 선택은 첫 번째 단계에서 이산 단위를 학습한 후 문맥화된 표현을 학습하는 이전 작업에서 좋은 결과를 가져왔습니다 [5]. 제품 양자화는 여러 코드북에서 양자화된 표현을 선택하고 이를 연결하는 것입니다. V개의 엔트리가 있는 G개의 코드북 또는 그룹이 주어지면, 각 코드북에서 하나의 엔트리를 선택하고 결과 벡터 e1, ..., ..., eG를 연결합니다. , eG를 연결하고 선형 변환 R d 7→ R f 를 적용하여 q ∈ R f 를 얻습니다.

검벨 소프트맥스는 이산 코드북 항목을 완전히 차별화할 수 있는 방식으로 선택할 수 있습니다[16, 24, 35]. 여기서는 직선 추정기[26]를 사용하고 G 하드 Gumbel 소프트맥스 연산[24]을 설정합니다. 특징 인코더 출력 z는 l ∈ R G×V 로짓에 매핑되며, 그룹 g에 대한 v 번째 코드북 항목을 선택할 확률은 다음과 같습니다.

여기서 τ는 음수가 아닌 온도, n = - 로그(- 로그(u)), u는 U(0, 1)의 균일한 샘플입니다. 포워드 패스 동안 코드워드 i는 i = argmaxj pg,j로 선택되며, 백워드 패스에서는 검벨 소프트맥스 출력의 실제 기울기가 사용됩니다.

## Training

<details>
<summary>en</summary>

To pre-train the model we mask a certain proportion of time steps in the latent feature encoder space (§ 3.1), similar to masked language modeling in BERT [9]. The training objective requires identifying the correct quantized latent audio representation in a set of distractors for each masked time step (§ 3.2) and the final model is fine-tuned on the labeled data (§ 3.3).
</details>

모델을 사전 훈련하기 위해 BERT [9]의 마스킹 언어 모델링과 유사하게 잠재 특징 인코더 공간에서 특정 비율의 시간 단계를 마스킹합니다(§ 3.1). 훈련 목표는 각 마스킹된 시간 단계에 대해 일련의 방해 요소에서 올바른 양자화된 잠재 오디오 표현을 식별하는 것이며(§ 3.2), 최종 모델은 라벨이 지정된 데이터에 대해 미세 조정됩니다(§ 3.3).

### Masking

<details>
<summary>en</summary>

We mask a proportion of the feature encoder outputs, or time steps before feeding them to the context network and replace them with a trained feature vector shared between all masked time steps; we do not mask inputs to the quantization module. To mask the latent speech representations output by the encoder, we randomly sample without replacement a certain proportion p of all time steps to be starting indices and then mask the subsequent M consecutive time steps from every sampled index; spans may overlap.
</details>

특징 인코더 출력의 일부 또는 시간 단계를 마스킹하여 컨텍스트 네트워크에 공급하기 전에 마스킹된 모든 시간 단계 간에 공유되는 훈련된 특징 벡터로 대체하고 양자화 모듈에 대한 입력은 마스킹하지 않습니다. 인코더가 출력하는 잠재 음성 표현을 마스킹하기 위해 모든 시간 단계의 특정 비율 p를 교체하지 않고 무작위로 샘플링하여 시작 인덱스가 된 다음 샘플링된 모든 인덱스에서 후속 M개의 연속 시간 단계를 마스킹합니다(스팬이 겹칠 수 있음).

### Objective

<details>
<summary>en</summary>

During pre-training, we learn representations of speech audio by solving a contrastive task Lm which requires to identify the true quantized latent speech representation for a masked time step within a set of distractors. This is augmented by a codebook diversity loss Ld to encourage the model to use the codebook entries equally often.

L = Lm + αLd

where α is a tuned hyperparameter.

Contrastive Loss. Given context network output ct centered over masked time step t, the model needs to identify the true quantized latent speech representation qt in a set of K + 1 quantized candidate representations ˜q ∈ Qt which includes qt and K distractors [23, 54]. Distractors are uniformly sampled from other masked time steps of the same utterance. The loss is defined as

Lm = − log exp(sim(ct, qt)/κ) P ˜q∼Qt exp(sim(ct, ˜q)/κ)

where we compute the cosine similarity sim(a, b) = a T b/kakkbk between context representations and quantized latent speech representations [19, 6].

Diversity Loss. The contrastive task depends on the codebook to represent both positive and negative examples and the diversity loss Ld is designed to increase the use of the quantized codebook representations [10]. We encourage the equal use of the V entries in each of the G codebooks by maximizing the entropy of the averaged softmax distribution l over the codebook entries for each

codebook p¯g across a batch of utterances; the softmax disribution does not contain the gumbel noise nor a temperature:2

Ld = 1 GV X G g=1 −H(¯pg) = 1 GV X G g=1 X V v=1 p¯g,v log ¯pg,v
</details>

사전 훈련 중에는 방해 요소 세트 내에서 마스크된 시간 단계에 대한 실제 양자화된 잠복 음성 표현을 식별해야 하는 대조 과제 Lm을 해결하여 음성 오디오의 표현을 학습합니다. 여기에는 코드북 다양성 손실 Ld가 추가되어 모델이 코드북 항목을 동일하게 자주 사용하도록 장려합니다.

L = Lm + αLd

여기서 α는 조정된 하이퍼파라미터입니다.

대비 손실. 마스크된 시간 단계 t를 중심으로 한 컨텍스트 네트워크 출력 ct가 주어지면, 모델은 qt와 K 개의 산만 요소를 포함하는 K + 1 양자화된 후보 표현 ˜q ∈ Qt 세트에서 실제 양자화된 잠복 음성 표현 qt를 식별해야 합니다[23, 54]. 방해 요소는 동일한 발화의 다른 마스크된 시간 단계로부터 균일하게 샘플링됩니다. 손실은 다음과 같이 정의됩니다.

여기서 컨텍스트 표현과 양자화된 잠재 음성 표현 사이의 코사인 유사성 sim(a, b) = a T b/kakkbk를 계산합니다[19, 6].

다양성 손실. 대조 작업은 코드북에 따라 양수 예제와 음수 예제를 모두 표현하며 다양성 손실 Ld는 양자화된 코드북 표현의 사용을 늘리기 위해 설계되었습니다[10]. 각 코드북 항목에 대한 평균 소프트맥스 분포의 엔트로피를 최대화하여 각 G 코드북에서 V 항목을 동등하게 사용하도록 권장합니다.

발화 일괄 처리에서 코드북 p¯g; 소프트맥스 배포에는 굼벨 노이즈나 온도가 포함되지 않습니다.2

Ld = 1 GV X G g=1 −H(¯pg) = 1 GV X G g=1 X V v=1 p¯g,v log ¯pg,v

### Fine-tuning

<details>
<summary>en</summary>

Pre-trained models are fine-tuned for speech recognition by adding a randomly initialized linear projection on top of the context network into C classes representing the vocabulary of the task [4]. For Librispeech, we have 29 tokens for character targets plus a word boundary token. Models are optimized by minimizing a CTC loss [14] and we apply a modified version of SpecAugment [41] by masking to time-steps and channels during training which delays overfitting and significantly improves the final error rates, especially on the Libri-light subsets with few labeled examples.
</details>

사전 학습된 모델은 작업의 어휘를 나타내는 C 클래스에 컨텍스트 네트워크 위에 무작위로 초기화된 선형 투영을 추가하여 음성 인식을 위해 미세 조정됩니다[4]. 라이브러스피치에는 문자 타깃을 위한 29개의 토큰과 단어 경계 토큰이 있습니다. 모델은 CTC 손실을 최소화하여 최적화되며[14], 훈련 중에 시간 단계와 채널에 마스킹을 적용하여 과적합을 지연시키고 특히 라벨링된 예가 적은 Libri-light 하위 집합에서 최종 오류율을 크게 개선하는 수정된 버전의 SpecAugment[41]를 적용합니다.

### Language Models and Decoding

<details>
<summary>en</summary>

We consider two types of language models (LM): a 4-gram model and a Transformer [3] trained on the Librispeech LM corpus. The Transformer LM is identical to [51] and contains 20 blocks, model dimension 1,280, inner dimension 6,144 and 16 attention heads. We tune the weights of the language model (interval [0, 5]) and a word insertion penalty ([−5, 5]) via Bayesian optimization3 : we run 128 trials with beam 500 for the 4-gram LM and beam 50 for the Transformer LM and choose the best set of weights according to performance on dev-other. Test performance is measured with beam 1,500 for the n-gram LM and beam 500 for the Transformer LM. We use the beam search decoder of [44].
</details>

우리는 두 가지 유형의 언어 모델(LM)을 고려합니다: 4그램 모델과 Librispeech LM 코퍼스로 훈련된 Transformer [3]. 트랜스포머 LM은 [51]과 동일하며 20개의 블록, 모델 크기 1,280, 내부 크기 6,144, 16개의 주의 헤드를 포함합니다. 베이지안 최적화3를 통해 언어 모델의 가중치(간격 [0, 5])와 단어 삽입 패널티([-5, 5])를 조정합니다. 4그램 LM의 경우 빔 500, Transformer LM의 경우 빔 50으로 128번의 실험을 실행하고 dev-other에서의 성능에 따라 최적의 가중치 세트를 선택합니다. 테스트 성능은 n-그램 LM의 경우 빔 1,500으로, Transformer LM의 경우 빔 500으로 측정합니다. 빔 검색 디코더는 [44]의 빔 검색 디코더를 사용합니다.

## Results

### Low-Resource Labeled Data Evaluation

<details>
<summary>en</summary>

We first evaluate our pre-trained models in settings where the amount of labeled data is limited to get a sense of how the representations learned on unlabeled data can improve low resource settings. If a pre-trained model captures the structure of speech, then it should require few labeled examples to fine-tune it for speech recognition. The models are pre-trained on the audio data of either Librispeech (LS-960) or LibriVox (LV-60k) and most results are obtained by decoding with a Transformer language model (Transf.); Appendix C shows results with no language model at all as well as with an n-gram language model.

The LARGE model pre-trained on LV-60k and fine-tuned on only 10 minutes of labeled data achieves a word error rate of 5.2/8.6 on the Librispeech clean/other test sets. Ten minutes of labeled data corresponds to just 48 recordings with an average length of 12.5 seconds. This demonstrates that ultra-low resource speech recognition is possible with self-supervised learning on unlabeled data.

Our approach of jointly learning discrete units and contextualized representations clearly improves over previous work which learned quantized audio units in a separate step [4], reducing WER by a about a third.

A recent iterative self-training approach [42] represents the state of the art on the clean 100 hour subset of Librispeech but it requires multiple iterations of labeling, filtering, and re-training. Our approach is simpler: we pre-train on the unlabeled data and fine-tune on the labeled data. On the 100 hour subset of Librispeech, their method achieves WER 4.2/8.6 on test-clean/other which compares to WER 2.3/5.0 with the LARGE model in a like for like setup, a relative WER reduction of 45%/42%.

When the LARGE model uses an order of magnitude less labeled data (10h labeled), then it still achieves WER 3.2/6.1, an error reduction of 24%/29% relative to iterative self-training. Using only a single hour of labeled data, the same model achieves WER 3.9/7.6 which improves on both test-clean and test-other by 7%/12% - with two orders of magnitude less labeled data. We note that the Librilight data splits contain both clean and noisy data leading to better accuracy on test-other compared to test-clean. Increasing model size reduces WER on all setups with the largest improvements on test-other (BASE vs. LARGE both on LS-960) and increasing the amount of unlabeled training data also leads to large improvements (LARGE LS-960 vs. LV-60k)
</details>

먼저 레이블이 지정된 데이터의 양이 제한된 환경에서 사전 학습된 모델을 평가하여 레이블이 지정되지 않은 데이터에서 학습된 표현이 리소스가 부족한 설정을 어떻게 개선할 수 있는지 파악합니다. 사전 훈련된 모델이 음성의 구조를 파악하고 있다면 음성 인식을 위해 미세 조정하는 데 라벨링된 예시가 거의 필요하지 않습니다. 이 모델은 라이브러스피치(LS-960) 또는 라이브러복스(LV-60k)의 오디오 데이터로 사전 훈련되었으며 대부분의 결과는 트랜스포머 언어 모델(Transf.)로 디코딩하여 얻었습니다. 부록 C는 언어 모델이 전혀 없는 경우와 n-그램 언어 모델을 사용한 결과를 보여줍니다.

LV-60k로 사전 학습하고 10분의 라벨링된 데이터로만 미세 조정된 LARGE 모델은 Librispeech 클린/기타 테스트 세트에서 5.2/8.6의 단어 오류율을 달성했습니다. 10분의 라벨링된 데이터는 평균 길이가 12.5초인 48개의 녹음에 해당합니다. 이는 레이블이 없는 데이터에 대한 자가 지도 학습을 통해 초저자원 음성 인식이 가능하다는 것을 보여줍니다.

개별 단위와 문맥화된 표현을 공동으로 학습하는 접근 방식은 양자화된 오디오 단위를 별도의 단계에서 학습하는 이전 작업[4]보다 확실히 개선되어 WER을 약 3분의 1로 줄였습니다.

최근의 반복적 자가 학습 접근법[42]은 깨끗한 100시간의 Librispeech 하위 집합에 대한 최신 기술이지만 라벨링, 필터링 및 재학습을 여러 번 반복해야 합니다. 우리의 접근 방식은 더 간단합니다. 라벨링되지 않은 데이터에 대해 사전 학습하고 라벨링된 데이터에 대해 미세 조정합니다. 이 방법은 100시간의 Librispeech 하위 집합에서 테스트-청소/기타에서 WER 4.2/8.6을 달성하여 유사 설정에서 LARGE 모델의 WER 2.3/5.0에 비해 45%/42%의 상대적 WER 감소를 보였습니다.

LARGE 모델이 훨씬 적은 레이블이 지정된 데이터(레이블이 지정된 10시간)를 사용하더라도 반복적인 자가 학습에 비해 24%/29% 오류가 감소한 WER 3.2/6.1을 달성할 수 있습니다. 동일한 모델이 단 1시간의 라벨링된 데이터만 사용할 경우, 라벨링된 데이터가 두 배나 적은 상태에서 테스트-클린과 테스트-기타 모두 7%/12% 개선된 WER 3.9/7.6을 달성합니다. Librilight 데이터 분할에는 클린 데이터와 노이즈 데이터가 모두 포함되어 있어 테스트-클린에 비해 테스트-기타의 정확도가 더 높다는 점에 주목할 필요가 있습니다. 모델 크기를 늘리면 모든 설정에서 WER이 감소하며 테스트-기타에서 가장 크게 개선되고(LS-960의 경우 BASE 대 LARGE), 라벨링되지 않은 학습 데이터의 양을 늘리면 큰 개선 효과가 나타납니다(LARGE LS-960 대 LV-60k).

### High-Resource Labeled Data Evaluation on Librispeech

<details>
<summary>en</summary>

In this section we evaluate the performance when large quantities of labeled speech are available to assess the effectiveness of our approach in a high resource setup. Specifically, we fine-tune the same models as before on the full 960 hours of labeled Librispeech: BASE and LARGE pre-trained on LS-960 as well as LARGE pre-trained on LV-60k.

Table 2 shows that our approach achieves WER 1.8/3.3 on test-clean/other on the full Librispeech benchmark. This is despite a weaker baseline architecture: supervised training of our architecture achieves WER 2.1/4.6 (LARGE - from scratch) compared to WER 1.9/4.1 for ContextNet [17], the baseline architecture of the state of the art [42]. We use a simple Transformer with CTC which does not perform as well as seq2seq models [51].

Note that the vocabulary of our acoustic model (characters) does not match the vocabulary of the LM (words) which delays feedback from the LM and is likely to be detrimental. Most recent work [51, 58, 17, 42] uses the better performing word pieces [50] for both models. Moreover, our result is achieved without any data balancing such as [42]. Finally, self-training is likely complimentary to pre-training and their combination may yield even better results. Appendix E presents a detailed error analysis of our pre-trained models in various labeled data setups.
</details>

이 섹션에서는 대량의 레이블이 지정된 음성을 사용할 수 있을 때의 성능을 평가하여 리소스가 많은 설정에서 접근 방식의 효과를 평가합니다. 구체적으로, 960시간 분량의 라벨링된 라이브러스피치 전체에 대해 이전과 동일한 모델을 미세 조정합니다: LS-960에서 사전 학습된 BASE 및 LARGE와 LV-60k에서 사전 학습된 LARGE에 대해 이전과 동일한 모델을 미세 조정합니다.

표 2는 전체 Librispeech 벤치마크에서 테스트-클린/기타에서 우리의 접근 방식이 WER 1.8/3.3을 달성했음을 보여줍니다. 이는 기본 아키텍처가 더 약함에도 불구하고 달성한 결과입니다. 저희 아키텍처의 지도 학습은 최첨단 기본 아키텍처인 ContextNet [17]의 WER 1.9/4.1에 비해 WER 2.1/4.6(LARGE - 처음부터)을 달성했습니다[42]. 저희는 seq2seq 모델[51]만큼 성능이 좋지 않은 CTC와 함께 간단한 트랜스포머를 사용합니다.

음향 모델(문자)의 어휘가 LM(단어)의 어휘와 일치하지 않아 LM의 피드백이 지연되고 해로울 수 있다는 점에 유의하세요. 가장 최근의 연구[51, 58, 17, 42]는 두 모델 모두에 더 성능이 좋은 단어 조각[50]을 사용합니다. 또한, [42]와 같은 데이터 밸런싱 없이도 이러한 결과를 얻을 수 있습니다. 마지막으로, 자가 학습은 사전 학습을 보완할 수 있으며, 이 둘을 함께 사용하면 더 나은 결과를 얻을 수 있습니다. 부록 E에는 다양한 라벨링 데이터 설정에서 사전 학습된 모델에 대한 자세한 오류 분석이 나와 있습니다.

### Phoneme Recognition on TIMIT

<details>
<summary>en</summary>

Next, we evaluate accuracy on TIMIT phoneme recognition by fine-tuning the pre-trained models on the labeled TIMIT training data. We fine-tune as for the 10 hour subset of Libri-light but do not use a language model. Table 3 shows that our approach can achieve a new state of the art on this dataset, reducing PER by a relative 23%/29% over the next best result on the dev/test sets. Appendix D shows an analysis of how the discrete latent speech representations related to phonemes. Other recent work on pre-training which evaluates on TIMIT includes [47] who solve multiple tasks to learn good representations of speech.
</details>

다음으로, 레이블이 지정된 TIMIT 학습 데이터에 대해 사전 학습된 모델을 미세 조정하여 TIMIT 음소 인식의 정확도를 평가합니다. 라이브러리 라이트의 10시간 하위 집합에 대해서는 미세 조정하지만 언어 모델은 사용하지 않습니다. 표 3은 이 데이터 세트에서 우리의 접근 방식이 개발/테스트 세트의 차선책에 비해 상대적으로 23%/29% 더 낮은 PER을 달성할 수 있음을 보여줍니다. 부록 D는 음소와 관련된 이산 잠재 음성 표현의 분석 결과를 보여줍니다. TIMIT로 평가하는 사전 훈련에 대한 다른 최근 연구로는 음성의 좋은 표현을 학습하기 위해 여러 과제를 해결하는 [47]이 있습니다.

### Ablations

<details>
<summary>en</summary>

A difference to previous work [5, 4] is that we quantize the latent audio representations only for the contrastive loss, i.e., when latents are used as targets, but not when the latents are input to the Transformer network. We motivate this choice by an ablating for which we adopt a reduced training setup to increase experimental turn around: we pre-train BASE on LS-960 for 250k updates with masking probability p = 0.075, fine-tune on train-10h for 60k updates on a single GPU with 640k samples per batch, or 40 sec of speech audio. We report the average WER and standard deviation on the concatenation of dev-clean and dev-other (dev PER) for three seeds of fine-tuning.

Table 4 shows that our strategy of continuous inputs with quantized targets (Baseline) performs best. Continuous latent speech representations retain more information to enable better context representations and quantizing the target representations leads to more robust training. Quantizing the latents both in the input and the targets performs least well, and explains the lower performance of prior work [5, 4]. Continuous targets reduce the effectiveness of self-supervised training since targets can capture detailed artifacts of the current sequence, e.g. speaker and background information, which make the task easier and prevent the model from learning general representations beneficial to speech recognition. The training accuracy of identifying the correct latent audio representation increases from 62% to 78.0% when switching from quantized to continuous targets. Continuous inputs and continuous targets perform second best but various attempts to improve it did not lead to better results (see Appendix F for this experiment and other ablations on various hyperparameters).
</details>

이전 작업[5, 4]과의 차이점은 대비 손실에 대해서만, 즉 잠재가 표적으로 사용될 때만 잠재 오디오 표현을 정량화하고 잠재가 트랜스포머 네트워크에 입력될 때는 정량화하지 않는다는 것입니다. 마스킹 확률 p = 0.075의 250만 업데이트에 대해 LS-960에서 BASE를 사전 훈련하고, 배치당 640만 샘플 또는 40초의 음성 오디오로 단일 GPU에서 60만 업데이트에 대해 train-10h에서 미세 조정하여 실험 회전율을 높이는 축소 훈련 설정을 채택함으로써 이러한 선택의 동기를 부여합니다. 세 가지 미세 조정 시드에 대한 dev-clean과 dev-other의 연결에 대한 평균 WER 및 표준 편차를 보고합니다(dev PER).

표 4는 정량화된 목표(베이스라인)를 지속적으로 입력하는 전략이 가장 우수한 성능을 보인다는 것을 보여줍니다. 연속적인 잠재 음성 표현은 더 많은 정보를 보유하여 더 나은 문맥 표현을 가능하게 하고, 목표 표현을 정량화하면 더 강력한 훈련으로 이어집니다. 입력과 목표 모두에서 잠재어를 정량화하는 것은 성능이 가장 낮으며, 이전 작업의 낮은 성능을 설명합니다[5, 4]. 연속 타깃은 화자 및 배경 정보와 같은 현재 시퀀스의 세부 아티팩트를 캡처하여 작업을 더 쉽게 만들고 모델이 음성 인식에 도움이 되는 일반적인 표현을 학습하지 못하게 하기 때문에 자기 지도 훈련의 효과를 감소시킵니다. 정량화된 타겟에서 연속 타겟으로 전환할 때 올바른 잠재 오디오 표현을 식별하는 훈련 정확도는 62%에서 78.0%로 증가합니다. 연속 입력과 연속 타겟은 두 번째로 좋은 성능을 보였지만 이를 개선하기 위한 다양한 시도에도 불구하고 더 나은 결과를 얻지 못했습니다(이 실험과 다양한 하이퍼파라미터에 대한 다른 절제법은 부록 F 참조).

## Conclusion

<details>
<summary>en</summary>

We presented wav2vec 2.0, a framework for self-supervised learning of speech representations which masks latent representations of the raw waveform and solves a contrastive task over quantized speech representations. Our experiments show the large potential of pre-training on unlabeled data for speech processing: when using only 10 minutes of labeled training data, or 48 recordings of 12.5 seconds on average, we achieve a WER of 4.8/8.2 on test-clean/other of Librispeech.

Our model achieves results which achieve a new state of the art on the full Librispeech benchmark for noisy speech. On the clean 100 hour Librispeech setup, wav2vec 2.0 outperforms the previous best result while using 100 times less labeled data. The approach is also effective when large amounts of labeled data are available. We expect performance gains by switching to a seq2seq architecture and a word piece vocabulary
</details>

저희는 원시 파형의 잠재적 표현을 마스킹하고 양자화된 음성 표현에 대한 대조 작업을 해결하는 음성 표현의 자가 지도 학습 프레임워크인 wav2vec 2.0을 발표했습니다. 실험 결과, 라벨링되지 않은 데이터에 대한 사전 학습이 음성 처리에 큰 잠재력을 가지고 있음을 보여주었습니다. 10분 분량의 라벨링된 학습 데이터 또는 평균 12.5초 분량의 48개 녹음만 사용했을 때 Librispeech의 테스트-클린/기타에서 4.8/8.2의 WER을 달성했습니다.

이 모델은 노이즈가 있는 음성에 대한 전체 Librispeech 벤치마크에서 새로운 기술 수준을 달성하는 결과를 얻었습니다. 깨끗한 100시간의 Librispeech 설정에서 wav2vec 2.0은 100배 적은 레이블 데이터를 사용하면서 이전 최고 결과를 능가하는 성능을 보였습니다. 이 접근 방식은 대량의 라벨링된 데이터를 사용할 수 있는 경우에도 효과적입니다. seq2seq 아키텍처와 단어 조각 어휘로 전환하면 성능이 향상될 것으로 기대합니다.

### Broader Impact

<details>
<summary>en</summary>

There are around 7,000 languages in the world and many more dialects. However, for most of them no speech recognition technology exists since current systems require hundreds or thousands of hours of labeled data which is hard to collect for most languages. We have shown that speech recognition models can be built with very small amounts of annotated data at very good accuracy. We hope our work will make speech recognition technology more broadly available to many more languages and dialects.
</details>

전 세계에는 약 7,000개의 언어와 수많은 방언이 존재합니다. 하지만 대부분의 언어에 대해 수집하기 어려운 수백, 수천 시간의 라벨링 데이터가 필요하기 때문에 현재 시스템에는 음성 인식 기술이 존재하지 않습니다. 저희는 아주 적은 양의 주석이 달린 데이터로도 음성 인식 모델을 매우 정확하게 구축할 수 있음을 보여주었습니다. 저희의 연구를 통해 더 많은 언어와 방언에 음성 인식 기술을 더 폭넓게 적용할 수 있기를 바랍니다.
