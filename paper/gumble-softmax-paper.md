# CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX

## ABSTRACT

Categorical variables are a natural choice for representing discrete structure in the world. However, stochastic neural networks rarely use categorical latent variables due to the inability to backpropagate through samples. In this work, we present an efficient gradient estimator that replaces the non-differentiable sample from a categorical distribution with a differentiable sample from a novel Gumbel-Softmax distribution. This distribution has the essential property that it can be smoothly annealed into a categorical distribution. We show that our Gumbel-Softmax estimator outperforms state-of-the-art gradient estimators on structured output prediction and unsupervised generative modeling tasks with categorical latent variables, and enables large speedups on semi-supervised classification.

범주형 변수는 세상의 불연속적인 구조를 표현하는 데 자연스러운 선택입니다. 그러나 확률적 신경망은 샘플을 통한 역전파가 불가능하기 때문에 범주형 잠재 변수를 거의 사용하지 않습니다. 이 연구에서는 범주형 분포에서 미분할 수 없는 표본을 새로운 검벨-소프트맥스 분포에서 미분할 수 있는 표본으로 대체하는 효율적인 기울기 추정기를 제시합니다. 이 분포는 범주형 분포로 부드럽게 어닐링할 수 있다는 본질적인 특성을 가지고 있습니다. 저희는 범주형 잠재 변수를 사용하는 구조화된 출력 예측 및 비지도 생성 모델링 작업에서 Gumbel-Softmax 추정기가 최첨단 기울기 추정기보다 성능이 뛰어나며, 준지도 분류에서 속도를 크게 향상시킬 수 있음을 보여줍니다.

## INTRODUCTION

Stochastic neural networks with discrete random variables are a powerful technique for representing distributions encountered in unsupervised learning, language modeling, attention mechanisms, and reinforcement learning domains. For example, discrete variables have been used to learn probabilistic latent representations that correspond to distinct semantic classes (Kingma et al., 2014), image regions (Xu et al., 2015), and memory locations (Graves et al., 2014; Graves et al., 2016). Discrete representations are often more interpretable (Chen et al., 2016) and more computationally efficient (Rae et al., 2016) than their continuous analogues.

However, stochastic networks with discrete variables are difficult to train because the backpropagation algorithm — while permitting efficient computation of parameter gradients — cannot be applied to non-differentiable layers. Prior work on stochastic gradient estimation has traditionally focused on either score function estimators augmented with Monte Carlo variance reduction techniques (Paisley et al., 2012; Mnih & Gregor, 2014; Gu et al., 2016; Gregor et al., 2013), or biased path derivative estimators for Bernoulli variables (Bengio et al., 2013). However, no existing gradient estimator has been formulated specifically for categorical variables. The contributions of this work are threefold:

1. We introduce Gumbel-Softmax, a continuous distribution on the simplex that can approximate categorical samples, and whose parameter gradients can be easily computed via the reparameterization trick.
2. We show experimentally that Gumbel-Softmax outperforms all single-sample gradient estimators on both Bernoulli variables and categorical variables.
3. We show that this estimator can be used to efficiently train semi-supervised models (e.g. Kingma et al. (2014)) without costly marginalization over unobserved categorical latent variables.

The practical outcome of this paper is a simple, differentiable approximate sampling mechanism for categorical variables that can be integrated into neural networks and trained using standard backpropagation.

이산 확률 변수를 사용하는 확률 신경망은 비지도 학습, 언어 모델링, 주의 메커니즘, 강화 학습 영역에서 발생하는 분포를 표현하는 강력한 기법입니다. 예를 들어, 이산 변수는 별개의 의미 클래스(Kingma 외, 2014), 이미지 영역(Xu 외, 2015), 기억 위치(Graves 외, 2014; Graves 외, 2016)에 해당하는 확률적 잠재 표현을 학습하는 데 사용되었습니다. 불연속적 표현은 연속적 표현보다 해석이 더 쉽고(Chen et al., 2016) 계산적으로 더 효율적입니다(Rae et al., 2016).

그러나 이산 변수가 있는 확률론적 네트워크는 역전파 알고리즘이 파라미터 기울기를 효율적으로 계산할 수 있지만 비분화 가능한 레이어에는 적용할 수 없기 때문에 훈련하기가 어렵습니다. 확률적 기울기 추정에 대한 기존 연구는 몬테카를로 분산 감소 기법으로 보강된 점수 함수 추정기(Paisley et al., 2012; Mnih & Gregor, 2014; Gu et al., 2016; Gregor et al., 2013) 또는 베르누이 변수에 대한 편향 경로 미분 추정기(Bengio et al., 2013)에 중점을 두었습니다. 그러나 범주형 변수를 위해 특별히 공식화된 기존 경사 추정기는 없습니다. 이 연구의 기여는 세 가지입니다:

1. 범주형 샘플을 근사화할 수 있고 매개변수 재매개화 트릭을 통해 매개변수 기울기를 쉽게 계산할 수 있는 심플렉스의 연속 분포인 Gumbel-Softmax를 소개합니다.
2. 베르누이 변수와 범주형 변수 모두에서 굼벨-소프트맥스가 모든 단일 샘플 기울기 추정기보다 우수한 성능을 보인다는 것을 실험적으로 보여줍니다.
3. 관측되지 않은 범주형 잠재 변수에 대해 비용이 많이 드는 한계화 없이 이 추정법을 사용하여 준지도 모형(예: Kingma et al. (2014))을 효율적으로 훈련할 수 있음을 보여줍니다.

이 백서의 실질적인 결과는 신경망에 통합하고 표준 역전파를 사용하여 훈련할 수 있는 범주형 변수에 대한 간단하고 차별적인 근사 샘플링 메커니즘입니다.

```
이산 확률 변수: 확률 변수가 얻을 수 있는 값의 범위가 무한하던, 유한다던 간에 값을 셀 수 있는 확률 변수를 말함.
예: 동전 0 ~ 100 번 던졌을 때 얻는 정수 값이 이산 확률 변수

연속 확률 변수: 확률 변수가 취할 수 있는 값이 특정 범위의 값으로, 값의 수가 무한한 확률 변수를 말함.
예: 체중, 온도, 키와 같이 연속적인 값 이라도 값이 무한하게 많아지는 것이 연속 확률 변수

이산 확률 분포: 이산 확률 변수를 가지는 분포
예: 이항분포

연속 확률 분포: 연속 확률 변수를 가지는 분포
예: 정규분포

```

## THE GUMBEL-SOFTMAX DISTRIBUTION

We begin by defining the Gumbel-Softmax distribution, a continuous distribution over the simplex that can approximate samples from a categorical distribution. Let z be a categorical variable with class probabilities π1, π2, ...πk. For the remainder of this paper we assume categorical samples are encoded as k-dimensional one-hot vectors lying on the corners of the (k − 1)-dimensional simplex, ∆k−1 . This allows us to define quantities such as the element-wise mean Ep[z] = [π1, ..., πk] of these vectors.
