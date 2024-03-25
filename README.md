# wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

## Abstract

### 기존 방식의 문제점

- 기존 STT에서 사용할만한 수준의 성능을 구현하기 위해선 못해도 수천시간의 전사된 음성이 필요로 함.
하지만 전 세계에 존재하는 7000개의 언어 중 대부분은 해당 조건을 맞추기 어려운 언어가 대부분.

- 기존 STT에선 2 stage로 학습이 진행 되었으며, 1-step으로 양자화된 표현을 학습, 2-step으로 전사된 문자와 매칭을 시키는 방식으로 학습을 진행해 왔었음.
그러다 보니 학습 파이프 라인이 복잡하고 학습이 불안정 해지는 문제가 존재함.

### 논문이 제안하는 방식

- 전사되지 않은 음성으로 부터도 음성의 특징을 학습할 수 있는 self supervised 방식의 PreTrain 기법을 소개함.

## Introduction

## Model

## Training

### Masking

### Objective
