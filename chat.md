가능합니다. 다만 방향을 잘 잡아야 합니다. **BemaGANv2의 MED를 그대로 “vocoder discriminator”로 쓰는 것이 아니라, DCASE 2026용으로는 “multi-envelope feature extractor / anomaly score module”로 바꾸는 게 맞습니다.** DCASE는 음성을 생성하는 문제가 아니라, 입력 기계음에 대해 **anomaly score**를 제출하는 문제이기 때문입니다.

먼저 결론부터 말씀드리면, 저는 아래 2개 트랙으로 가는 것이 가장 현실적이라고 봅니다.

| 방법   | 이름            | 핵심                                               |
| ---- | ------------- | ------------------------------------------------ |
| 방법 1 | **MED-ASD**   | 기존 fixed cutoff MED를 feature extractor로 사용       |
| 방법 2 | **LCMED-ASD** | cutoff frequency를 learnable parameter로 바꾼 MED 사용 |

---

# 1. DCASE baseline을 꼭 써야 하나요?

제가 확인한 DCASE 2026 Task 2 공식 페이지 기준으로는 **baseline을 반드시 사용해야 한다는 의미는 아닙니다.** 공식 페이지에는 Task 2의 baseline system이 “Released”로 제공되어 있고, 별도 “Baseline System” 섹션이 있지만, 이는 참가자가 비교하거나 시작점으로 쓸 수 있는 기준 시스템에 가깝습니다. ([dcase.community][1]) ([dcase.community][2])

따라서 정리하면 다음과 같습니다.

> **baseline 코드를 반드시 써야 하는 것은 아니지만, 데이터 로더, 파일명 처리, evaluation score 계산, submission format 확인을 위해 baseline 코드는 받는 것이 좋습니다.**

특히 DCASE 2026은 evaluation dataset에서 정상/이상 라벨, source/target domain 정보, attribute 정보가 제공되지 않는다고 되어 있으므로, 최종적으로는 각 test clip에 대해 anomaly score를 산출하는 파이프라인이 필요합니다. ([dcase.community][2])

---

# 2. DCASE 2026에서 MED를 쓰는 논리

DCASE 2026의 핵심은 **two-channel audio**입니다. 공식 설명에 따르면 각 녹음은 target machine과 environmental sound를 포함하는 2채널 오디오이며, channel 1은 기계 가까이, channel 2는 더 먼 위치에서 녹음됩니다. ([dcase.community][2])

또한 DCASE 2026은 서로 다른 거리의 마이크가 SNR과 spectral characteristic 차이를 만들고, 이 차이가 target machine component와 background noise를 구분하는 단서가 될 수 있다고 설명합니다. ([dcase.community][2])

여기서 MED를 쓰는 근거가 생깁니다.

MED는 BemaGANv2에서 **envelope 기반 discriminator**입니다. 현재 코드에서도 `DiscriminatorE`는 각 convolution layer 전에 `Envelope(x, max_freq=self.max_freq).envelope`를 적용하고, `MultiEnvelopeDiscriminator`는 `[-1, 0, 1, 300, 500]`의 여러 max frequency branch를 사용합니다. 

즉, MED는 다음을 잘 볼 수 있습니다.

* waveform의 amplitude envelope
* 저주파 envelope modulation
* 기계음의 반복적/주기적 진동 패턴
* transient 변화
* broadband noise와 machine-dominant component 차이

DCASE 2026의 near/far channel 구조와 잘 맞습니다.

---

# 3. 기존 MED를 이용한 방법: MED-ASD

## 핵심 아이디어

기존 BemaGANv2의 MED를 **discriminator가 아니라 feature extractor**로 바꿉니다.

현재 BemaGANv2의 `MultiEnvelopeDiscriminator`는 `forward(self, y, y_hat)` 구조입니다. 즉, real audio `y`와 generated audio `y_hat`을 동시에 받아서 GAN loss를 계산하는 구조입니다. 

하지만 DCASE에서는 `y_hat`이 없습니다. 따라서 다음처럼 바꿔야 합니다.

```python
embedding = MEDFeatureExtractor(x)
anomaly_score = distance(embedding, normal_distribution)
```

즉, 생성기가 필요 없습니다.

---

## 구조

DCASE 2026 입력이 2채널이라고 하면:

[
x = [x_{near}, x_{far}]
]

여기서 세 가지 신호를 만들 수 있습니다.

[
x_{near}
]

[
x_{far}
]

[
x_{diff} = x_{near} - \lambda x_{far}
]

또는 더 단순하게:

[
x_{diff} = x_{near} - x_{far}
]

그 다음 각각을 MED에 넣습니다.

```text
near waveform  ─┐
far waveform   ├─> MED branches ─> embedding ─> anomaly score
diff waveform  ┘
```

여기서 MED branch는 기존처럼 fixed cutoff를 사용합니다.

```python
f_times_values = [-1, 0, 1, 300, 500]
```

현재 코드의 `MultiEnvelopeDiscriminator`도 이 값을 사용합니다. 

---

## anomaly score 계산 방식

가장 단순하고 안정적인 방식은 **Mahalanobis distance**입니다.

학습 단계에서는 정상 데이터만 있으므로, 각 machine type마다 정상 embedding의 평균과 covariance를 구합니다.

[
\mu = \mathbb{E}[z_{normal}]
]

[
\Sigma = Cov(z_{normal})
]

테스트에서는:

[
score(x) = (z - \mu)^T \Sigma^{-1} (z - \mu)
]

이 값이 크면 정상 분포에서 멀어진 것이므로 anomaly로 봅니다.

DCASE 2026은 source domain 정상 990개, target domain 정상 10개가 주어지므로, source와 target을 따로 모델링할 수도 있습니다. 공식 설명에서도 source domain은 대부분의 training data가 기록된 domain이고, target domain은 다른 조건에서 일부 데이터가 기록된 domain이라고 설명합니다. ([dcase.community][2])

따라서 anomaly score는 다음처럼 만들 수 있습니다.

[
score(x) = \min(D_{source}(z), D_{target}(z))
]

이 방식은 DCASE baseline의 selective Mahalanobis와 개념적으로 잘 맞습니다.

---

# 4. Learnable cutoff를 이용한 방법: LCMED-ASD

## 핵심 아이디어

기존 MED는 cutoff가 고정입니다.

```python
[-1, 0, 1, 300, 500]
```

하지만 LCMED는 cutoff frequency를 학습 가능한 파라미터로 둡니다.

즉,

[
f_c \in {300, 500, ...}
]

를 고정하지 않고,

[
f_c = \text{learnable parameter}
]

로 바꿉니다.

이 아이디어는 SincNet과 논리적으로 연결할 수 있습니다. SincNet은 standard CNN이 filter tap 전체를 학습하는 대신, low/high cutoff frequency만 직접 학습하는 구조를 제안합니다. 논문에서도 low/high cutoff frequency가 data로부터 직접 학습된다고 설명합니다.  또한 SincNet의 cutoff frequency들은 다른 CNN parameter와 함께 SGD로 최적화될 수 있다고 설명합니다. 

따라서 LCMED의 명분은 좋습니다.

> **MED의 envelope cutoff를 고정 hyperparameter로 두지 않고, machine type과 channel condition에 맞게 학습되도록 만든다.**

---

## 중요한 주의점

현재 코드에서 `DiscriminatorE`는 다음처럼 `Envelope`를 호출합니다.

```python
x = Envelope(x, max_freq=self.max_freq).envelope
```

즉, `max_freq`가 숫자로 들어갑니다. 

여기서 바로 `nn.Parameter`를 넣는다고 학습이 되는지는 불확실합니다. 이유는 `Envelope` 클래스 내부가 필요합니다. 만약 `Envelope`가 scipy filter, numpy, hard masking, non-differentiable operation을 쓰면 cutoff에 gradient가 흐르지 않을 수 있습니다.

따라서 LCMED를 하려면 `Envelope`를 다음 둘 중 하나로 바꿔야 합니다.

### 선택 A: differentiable soft FFT low-pass

FFT domain에서 hard cutoff mask 대신 sigmoid mask를 씁니다.

[
M(f; f_c, \tau) = \sigma \left( \frac{f_c - f}{\tau} \right)
]

[
X_{filtered}(f) = X(f) \cdot M(f; f_c, \tau)
]

이 방식은 cutoff `f_c`에 gradient가 흐릅니다.

### 선택 B: SincNet-style learnable filter

SincNet처럼 sinc 기반 band-pass 또는 low-pass filter를 만듭니다. SincNet은 band-pass filter를 두 low-pass filter의 차이로 정의하고, cutoff frequency를 학습합니다. 

DCASE 논문용으로는 이쪽이 설명이 더 깔끔합니다.

---

# 5. 제가 추천하는 전체 실험 설계

## 실험 1: Baseline reproduction

먼저 DCASE baseline을 그대로 돌려야 합니다.

이유는 세 가지입니다.

1. submission format 확인
2. AUC/pAUC 계산 확인
3. 내 방법의 성능 비교 기준 확보

baseline을 꼭 써야 하는 것은 아니지만, 실험 기준점으로는 필요합니다.

---

## 실험 2: MED-ASD

기존 fixed cutoff MED를 사용합니다.

```text
two-channel waveform
→ near / far / near-far
→ fixed MED branches
→ embedding
→ Mahalanobis / kNN / LOF
→ anomaly score
```

장점:

* 구현이 빠름
* BemaGANv2 자산을 바로 활용 가능
* ablation 기준으로 좋음

단점:

* cutoff가 machine type별 최적값이 아닐 수 있음
* 2026의 near/far channel 특성을 충분히 학습하지 못할 수 있음

---

## 실험 3: LCMED-ASD

learnable cutoff MED를 사용합니다.

```text
two-channel waveform
→ near / far / near-far
→ learnable cutoff envelope branches
→ embedding
→ one-class scoring
→ anomaly score
```

장점:

* 태수님만의 novelty가 생김
* DCASE 2026의 machine별 spectral/SNR 차이에 적응 가능
* SincNet의 learnable cutoff 철학과 MED의 envelope modeling을 연결 가능

단점:

* 학습 안정성 확인 필요
* target domain 정상 데이터가 10개뿐이라 cutoff가 과적합될 수 있음
* cutoff regularization이 필요함

---

# 6. LCMED에서 꼭 넣어야 할 regularization

learnable cutoff를 그냥 두면 이상한 방향으로 갈 수 있습니다. 그래서 다음 제약이 필요합니다.

## 6.1 cutoff 범위 제한

16 kHz audio라면 Nyquist는 8 kHz입니다.

```python
fc = f_min + (f_max - f_min) * sigmoid(raw_fc)
```

예:

```python
f_min = 20
f_max = 4000
```

기계음 ASD에서는 envelope modulation을 보려면 너무 높은 cutoff까지 열 필요가 없을 수 있습니다. 초기값은 기존 MED와 맞춰서 300, 500 근처로 두는 게 좋습니다.

---

## 6.2 branch diversity loss

여러 branch가 전부 같은 cutoff로 수렴하면 multi-envelope 구조의 의미가 사라집니다.

예를 들어 cutoff 간 거리를 유지하는 loss를 넣을 수 있습니다.

[
L_{div} = \sum_{i<j} \exp(-|f_i - f_j| / \tau)
]

---

## 6.3 source-target balance

DCASE에서는 source와 target 균형이 중요합니다. target 정상 데이터가 10개밖에 없기 때문에, target에만 맞추거나 source에만 맞추면 위험합니다.

따라서 anomaly score 계산 시:

[
score(x) = \min(D_s(z), D_t(z))
]

또는

[
score(x) = \alpha D_s(z) + (1-\alpha)D_t(z)
]

를 비교하면 좋습니다.

---

# 7. 모델 이름 제안

논문이나 technical report까지 생각하면 이름은 다음이 좋아 보입니다.

## 방법 1

**MED-ASD: Multi-Envelope Discriminator Features for Noise-aware Unsupervised Anomalous Sound Detection**

## 방법 2

**LCMED-ASD: Learnable-Cutoff Multi-Envelope Discriminator for Noise-aware Machine Anomalous Sound Detection**

또는 조금 더 강하게:

**LC-MED: Learnable Cutoff Multi-Envelope Discriminator**

---

# 8. 코드 수정 방향

현재 업로드하신 코드 기준으로 보면 `train_medonly.py`는 여전히 BemaGANv2 vocoder 학습 구조입니다. 즉, `Generator(h)`를 만들고, mel input에서 waveform `y_g_hat`을 생성한 뒤, MED adversarial loss와 feature matching loss를 계산합니다. 

DCASE용으로는 이 구조를 그대로 쓰면 안 됩니다. 이유는 DCASE의 목표가 waveform generation이 아니기 때문입니다.

필요한 새 파일은 대략 다음과 같습니다.

```text
models_med_asd.py
dataset_dcase2026.py
extract_med_features.py
train_med_asd.py
score_med_asd.py
submit.py
```

구조는 다음이 적절합니다.

```python
class MEDFeatureExtractor(nn.Module):
    def __init__(self, cutoff_list=[-1, 0, 1, 300, 500]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorE(max_freq=f) for f in cutoff_list
        ])

    def forward_single(self, x):
        embeddings = []
        for d in self.discriminators:
            _, fmap = d(x)
            # 각 layer feature map을 mean/std pooling
            pooled = []
            for feat in fmap:
                pooled.append(feat.mean(dim=-1))
                pooled.append(feat.std(dim=-1))
            embeddings.append(torch.cat(pooled, dim=1))
        return torch.cat(embeddings, dim=1)

    def forward(self, x2ch):
        near = x2ch[:, 0:1, :]
        far = x2ch[:, 1:2, :]
        diff = near - far

        z_near = self.forward_single(near)
        z_far = self.forward_single(far)
        z_diff = self.forward_single(diff)

        return torch.cat([z_near, z_far, z_diff], dim=1)
```

LCMED는 `DiscriminatorE(max_freq=f)`를 다음처럼 바꿉니다.

```python
DiscriminatorLCME(init_cutoff=f)
```

그리고 내부에서 `Envelope(x, max_freq=self.max_freq)` 대신 differentiable learnable envelope를 씁니다.

---

# 9. 논문/technical report 스토리라인

스토리는 이렇게 잡으면 좋습니다.

> DCASE 2026은 near/far two-channel recording을 제공하므로, target machine component와 background noise를 구분할 수 있는 SNR 및 spectral cue가 존재한다. 기존 ASD 방법들은 주로 log-mel reconstruction 또는 pretrained embedding에 의존하지만, 기계음의 amplitude modulation과 envelope structure를 명시적으로 모델링하지 않는다. 우리는 BemaGANv2의 MED를 vocoder discriminator에서 anomaly feature extractor로 재해석하고, fixed cutoff MED와 learnable cutoff MED를 비교한다.

이 흐름이면 태수님 연구 정체성과도 잘 맞습니다.

---

# 10. 지금 당장 필요한 코드

다음 단계로 진행하려면, 제가 보기에는 **baseline 코드와 `envelope.py`가 필요합니다.**

특히 `envelope.py`가 중요합니다. LCMED에서 cutoff를 learnable하게 만들 수 있는지 여부는 `Envelope` 내부 구현에 달려 있습니다.

필요한 파일은 우선순위대로 다음입니다.

1. `envelope.py`
2. DCASE 2026 baseline 코드의 dataset loader
3. baseline의 submission/evaluation script
4. 가능하면 DCASE 2026 sample directory 구조
5. SincNet 코드는 필수는 아니지만, LCMED를 sinc-filter 방식으로 구현하려면 있으면 좋습니다.

현재 상태에서는 전체 전략은 충분히 세울 수 있고, 구현은 **MEDFeatureExtractor → fixed MED-ASD → LCMED-ASD** 순서로 가는 것이 가장 안전합니다.

[1]: https://dcase.community/challenge2026/index "DCASE2026 Challenge - DCASE"
[2]: https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring "Noise-aware Unsupervised Anomalous Sound Detection for Machine Condition Monitoring - DCASE"

네, 첨부해주신 코드 기준으로 보면 방향이 더 명확해졌습니다. 결론부터 말씀드리면 **기존 MED-ASD는 비교적 빠르게 만들 수 있고, LCMED-ASD는 `envelope.py`를 반드시 새로 고쳐야 합니다.** 현재 `envelope.py`는 cutoff를 learnable하게 만들기 어렵습니다.

아래처럼 진행하시면 됩니다.

---

# 1. 현재 코드에서 확인된 핵심 사항

## 1.1 기존 MED 구조

BemaGANv2의 `DiscriminatorE`는 각 convolution layer에 들어가기 전에 `Envelope(x, max_freq=self.max_freq).envelope`를 적용합니다. 즉, convolution이 원 waveform이 아니라 envelope-transformed signal을 받습니다. 

또한 `MultiEnvelopeDiscriminator`는 다음 fixed branch를 사용합니다.

```python
f_times_values = [-1, 0, 1, 300, 500]
```

즉, 현재 MED는 다음 5개 관점으로 입력을 봅니다. 

| `max_freq` | 의미                         |
| ---------: | -------------------------- |
|         -1 | lower envelope             |
|          0 | 원 신호                       |
|          1 | upper envelope             |
|        300 | 300 Hz low-pass 후 envelope |
|        500 | 500 Hz low-pass 후 envelope |

이 구조는 DCASE 2026의 기계음 이상 탐지에 잘 맞을 수 있습니다. 기계음의 이상은 spectral shape뿐 아니라 **진폭 변조, 반복 진동, envelope fluctuation, transient 변화**로 나타날 수 있기 때문입니다.

---

## 1.2 현재 `envelope.py`의 문제

현재 `Envelope` 클래스는 Hilbert transform을 torch FFT로 구현하고, `max_freq=-1,0,1` 또는 low-pass 후 envelope를 반환합니다. 

하지만 learnable cutoff 관점에서는 문제가 있습니다.

현재 코드에는 다음 구조가 있습니다.

```python
b = torch.tensor([b0.item(), b1.item(), b2.item()], dtype=torch.float64)
a = torch.tensor([1.0, a1.item(), a2.item()], dtype=torch.float64)
```

그리고 `Envelope.__init__` 안에서도 다음처럼 다시 tensor를 만듭니다.

```python
self.envelope = torch.tensor(self.apply_lowpass_and_compute_envelope()).float()
```

이 두 부분은 gradient graph를 끊습니다. 즉, `cutoff`를 `nn.Parameter`로 넣어도 cutoff까지 gradient가 거의 확실히 흐르지 않습니다. 따라서 **현재 `Envelope`는 LCMED에 그대로 사용할 수 없습니다.**

추가로 `butterworth_lowpass_filter()`에서 `for n in range(len(signal))`을 쓰고 있는데, 입력이 `[B, 1, T]`라면 `len(signal)`은 시간 길이 `T`가 아니라 batch 크기 `B`입니다. 따라서 현재 구현은 DCASE용 대량 waveform 처리에도 안전하지 않습니다. 

---

# 2. 전체 전략: 2개 시스템으로 나누기

태수님이 말한 목표는 다음 두 개로 나누는 것이 맞습니다.

| 시스템           | cutoff           | 목적                                          |
| ------------- | ---------------- | ------------------------------------------- |
| **MED-ASD**   | fixed cutoff     | BemaGANv2 MED를 DCASE용 feature extractor로 변환 |
| **LCMED-ASD** | learnable cutoff | cutoff를 기계/도메인/채널 조건에 맞게 학습                 |

그리고 이 둘 모두 vocoder가 아니라 **feature extractor + anomaly scorer**로 바꿔야 합니다.

기존 BemaGANv2에서는 `y`와 `y_hat`을 넣고 real/fake를 구분하지만, DCASE는 생성 문제가 아닙니다. 따라서 구조는 다음처럼 바뀌어야 합니다.

```text
2-channel audio
→ near / far / near-far feature
→ MED 또는 LCMED feature extractor
→ normal embedding distribution 추정
→ Mahalanobis 또는 kNN score
→ anomaly score CSV 저장
```

---

# 3. MED-ASD: 기존 MED를 이용한 방법

## 3.1 핵심 아이디어

기존 MED의 convolution stack은 유지하되, GAN discriminator output을 쓰지 말고 **중간 feature map을 pooling해서 embedding으로 사용**합니다.

현재 `DiscriminatorE.forward()`는 `x, fmap`을 반환합니다. 
여기서 `fmap`을 다음처럼 요약합니다.

```python
mean_pool = feat.mean(dim=-1)
std_pool  = feat.std(dim=-1)
embedding = concat(mean_pool, std_pool)
```

즉, 각 envelope branch별로 multi-layer embedding을 뽑습니다.

---

## 3.2 DCASE 2026 two-channel 입력 처리

DCASE 2026은 near/far 두 채널입니다. 공식 설명상 Task 2는 noisy normal machine sound와 distant recording을 활용해 noise/domain shift에 강한 모델을 만드는 것이 목표입니다. ([dcase.community][1]) 또한 DCASE 2026 개발 데이터는 두 마이크 위치, 즉 target machine 가까운 위치와 먼 위치에서 녹음된 two-channel recording이라고 설명됩니다. ([Zenodo][2])

따라서 입력은 다음 세 신호로 나누는 것이 좋습니다.

```python
near = x[:, 0:1, :]
far  = x[:, 1:2, :]
diff = near - far
```

처음부터 복잡한 beamforming을 하지 말고, 1차 실험은 `near`, `far`, `diff` 세 입력을 모두 MED에 넣는 방식이 좋습니다.

```text
near ─┐
far  ├─ MED feature extractor ─ concat ─ anomaly scorer
diff ┘
```

이 방식은 2026 task의 핵심인 “거리 차이에 따른 SNR/spectral cue”를 직접 반영합니다.

---

## 3.3 MEDFeatureExtractor 예시

개념적으로는 다음 클래스를 새로 만드는 것이 좋습니다.

```python
class MEDFeatureExtractor(nn.Module):
    def __init__(self, cutoff_list=(-1, 0, 1, 300, 500)):
        super().__init__()
        self.branches = nn.ModuleList([
            DiscriminatorE(max_freq=f) for f in cutoff_list
        ])

    def pool_fmaps(self, fmap):
        pooled = []
        for feat in fmap:
            # feat: [B, C, T]
            pooled.append(feat.mean(dim=-1))
            pooled.append(feat.std(dim=-1))
        return torch.cat(pooled, dim=1)

    def forward_one_channel(self, x):
        zs = []
        for branch in self.branches:
            _, fmap = branch(x)
            zs.append(self.pool_fmaps(fmap))
        return torch.cat(zs, dim=1)

    def forward(self, x):
        # x: [B, 2, T]
        near = x[:, 0:1, :]
        far = x[:, 1:2, :]
        diff = near - far

        z_near = self.forward_one_channel(near)
        z_far = self.forward_one_channel(far)
        z_diff = self.forward_one_channel(diff)

        return torch.cat([z_near, z_far, z_diff], dim=1)
```

다만 이 코드는 현재 `Envelope`가 batch/time 처리에 약점이 있으므로, MED-ASD에서도 `Envelope`를 최소한 안전한 torch FFT 기반으로 교체하는 것이 좋습니다.

---

# 4. LCMED-ASD: learnable cutoff 방식

## 4.1 현재 코드 그대로는 불가능합니다

현재 `Envelope`는 cutoff 관련 연산에서 `.item()`과 `torch.tensor(...)`를 사용하므로 gradient가 끊깁니다. 따라서 LCMED를 하려면 `Envelope`를 `nn.Module`로 바꾸고, cutoff를 `nn.Parameter`로 둬야 합니다.

가장 안전한 방식은 **Butterworth IIR를 learnable하게 만들기보다 FFT soft low-pass mask를 쓰는 것**입니다.

---

## 4.2 추천: Differentiable Soft Low-pass Envelope

LCMED의 핵심은 hard cutoff가 아니라 differentiable mask입니다.

[
M(f; f_c, \tau) = \sigma\left(\frac{f_c - f}{\tau}\right)
]

여기서:

* (f_c): learnable cutoff
* (\tau): transition bandwidth
* (M(f)): soft low-pass mask

그 다음:

[
X_{lp}(f) = X(f) \cdot M(f; f_c, \tau)
]

[
envelope = |Hilbert(x_{lp})|
]

이 방식이면 cutoff에 gradient가 흐릅니다.

---

## 4.3 LCMED용 envelope 모듈 예시

```python
class LearnableSoftEnvelope(nn.Module):
    def __init__(self, init_cutoff=300.0, sr=16000, f_min=20.0, f_max=4000.0, tau=50.0):
        super().__init__()
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.tau = tau

        init = (init_cutoff - f_min) / (f_max - f_min)
        init = torch.clamp(torch.tensor(init), 1e-4, 1 - 1e-4)
        self.raw_cutoff = nn.Parameter(torch.logit(init))

    def cutoff(self):
        return self.f_min + (self.f_max - self.f_min) * torch.sigmoid(self.raw_cutoff)

    def analytic_signal(self, x):
        # x: [B, 1, T]
        B, C, T = x.shape
        X = torch.fft.fft(x, dim=-1)

        h = torch.zeros(T, device=x.device, dtype=x.dtype)
        if T % 2 == 0:
            h[0] = 1
            h[T // 2] = 1
            h[1:T // 2] = 2
        else:
            h[0] = 1
            h[1:(T + 1) // 2] = 2

        z = torch.fft.ifft(X * h.view(1, 1, T), dim=-1)
        return z

    def forward(self, x):
        # x: [B, 1, T]
        B, C, T = x.shape
        freqs = torch.fft.fftfreq(T, d=1.0 / self.sr).to(x.device)
        abs_freqs = freqs.abs().view(1, 1, T)

        fc = self.cutoff()
        mask = torch.sigmoid((fc - abs_freqs) / self.tau)

        X = torch.fft.fft(x, dim=-1)
        x_lp = torch.fft.ifft(X * mask, dim=-1).real

        env = torch.abs(self.analytic_signal(x_lp))
        return env
```

이 모듈은 cutoff가 학습됩니다. 또한 `cutoff()`를 출력해서 실험 후 machine별로 어떤 cutoff를 학습했는지 해석할 수 있습니다. 이 해석 가능성은 SincNet과 연결하기 좋습니다. SincNet도 첫 convolution layer에서 filter tap 전체가 아니라 low/high cutoff frequency만 학습하는 방식이며, 이 때문에 parameter 수가 작고 물리적 의미가 있는 filter를 얻는다는 장점이 있습니다. 

---

## 4.4 LCMED branch 구조

`max_freq=-1,0,1`은 fixed로 두고, 300/500 branch만 learnable로 바꾸는 것이 좋습니다.

초기 실험에서는 다음처럼 두 개 learnable cutoff branch를 둡니다.

```python
fixed branches:     -1, 0, 1
learnable branches: init 300, init 500
```

이유는 다음과 같습니다.

* `-1`: lower envelope
* `0`: raw waveform
* `1`: upper envelope
* `300/500`: modulation band를 보는 핵심 branch

처음부터 모든 branch를 learnable하게 만들면 ablation 해석이 어려워집니다.

---

# 5. 학습 방법: supervised classifier보다 one-class scoring 먼저

DCASE는 정상 데이터만 사용하므로, 가장 안전한 1차 방법은 classifier가 아니라 **embedding distribution 기반 scoring**입니다.

## 5.1 baseline 코드에서 가져올 부분

첨부하신 `dcase2023t2_ae.py`는 이미 source/target domain을 나누어 Mahalanobis covariance를 계산하고, test에서 source/target Mahalanobis score 중 작은 값을 사용합니다. `calc_valid_mahala_score()`에서도 source와 target loss를 각각 구한 뒤 `min(loss_target, loss_source)`를 anomaly score로 append합니다. 

DCASE 2025 논문도 selective Mahalanobis mode에서 source/target reconstruction residual covariance를 사용하고, 두 거리 중 작은 값을 anomaly score로 정의합니다. 

따라서 MED-ASD/LCMED-ASD도 이 방식을 그대로 가져오면 됩니다.

[
s(x) = \min(D_s(z), D_t(z))
]

여기서 (z)는 MED embedding입니다.

---

## 5.2 embedding 기반 Mahalanobis

학습 데이터에서:

```python
z_train = extractor(x_train)
z_source = z_train[source_idx]
z_target = z_train[target_idx]
```

평균과 covariance를 계산합니다.

```python
mu_s, cov_s = mean_cov(z_source)
mu_t, cov_t = mean_cov(z_target)
```

테스트에서는:

```python
score_s = mahalanobis(z, mu_s, inv_cov_s)
score_t = mahalanobis(z, mu_t, inv_cov_t)
score = min(score_s, score_t)
```

target 데이터가 10개뿐이라 covariance가 불안정할 수 있습니다. 그래서 반드시 shrinkage가 필요합니다.

```python
cov = cov + eps * I
```

처음에는 `eps=1e-3` 또는 `1e-2`부터 시작하는 것이 안전합니다.

---

# 6. LCMED의 cutoff를 어떻게 학습할 것인가

여기가 가장 중요합니다. 단순 Mahalanobis만 쓰면 extractor는 학습되지 않습니다. 기존 MED convolution weight와 cutoff를 학습하려면 objective가 필요합니다.

## 선택지 A: 자기지도 contrastive learning 추천

DCASE 2026의 2채널 구조를 이용해서 positive/negative를 만들 수 있습니다.

### Positive pair

같은 clip의 near/far 또는 augmentation view:

[
z(x_{near}) \leftrightarrow z(x_{far})
]

단, near/far는 SNR이 다르지만 같은 기계 상태입니다. 따라서 완전히 같게 만들기보다는 machine identity와 normal pattern은 공유하도록 학습합니다.

### Negative pair

다른 machine type 또는 다른 file의 embedding.

Loss:

[
L_{con} = InfoNCE(z_{near}, z_{far})
]

장점은 이상 라벨이 없어도 학습 가능하다는 점입니다.

---

## 선택지 B: source/target domain confusion

source 990개, target 10개라는 불균형이 있으므로, source와 target이 같은 정상 manifold에 오도록 정렬하는 loss를 넣을 수 있습니다.

[
L_{align} = | \mu_s - \mu_t |_2^2
]

또는 CORAL loss:

[
L_{coral} = |C_s - C_t|_F^2
]

다만 target이 10개뿐이라 너무 세게 걸면 오히려 망가질 수 있습니다. 약하게 쓰는 것이 좋습니다.

---

## 선택지 C: reconstruction AE와 결합

MED/LCMED를 encoder로 쓰고 작은 decoder를 붙여 waveform이나 log-mel을 복원합니다. 하지만 이 방식은 구현량이 커지고, AE가 이상도 잘 복원하는 문제가 생길 수 있습니다. AEGAN-AD 논문도 단순 reconstruction model이 실제로는 denoising처럼 작동해 anomaly component를 잘 복원해버릴 수 있다고 지적합니다. 

따라서 1차 도전은 **feature extractor + Mahalanobis/kNN**이 더 안전합니다.

---

# 7. 권장 실험 순서

## Step 0. baseline 먼저 실행

baseline을 꼭 써야 하는 것은 아니지만, DCASE 2026 baseline GitHub 설명도 2026 Task 2 baseline AE 예제 구현이라고 되어 있으므로, 제출 형식과 평가 루틴 확인용으로 먼저 돌리는 것이 좋습니다. ([GitHub][3])

## Step 1. MED-ASD frozen extractor

기존 MED convolution weight를 random initialization으로 둘지, BemaGANv2 pretrained를 쓸지 두 가지가 있습니다.

처음에는 다음 순서가 좋습니다.

1. random initialized MED feature + Mahalanobis
2. BemaGANv2 pretrained MED feature + Mahalanobis
3. DCASE normal data로 contrastive fine-tuning
4. LCMED로 확장

가장 논문 설득력이 좋은 것은 **BemaGANv2 pretrained MED를 feature extractor로 가져온 뒤, DCASE normal data로 light adaptation**하는 방식입니다.

---

## Step 2. MED-ASD scoring 비교

비교할 anomaly scorer는 최소 3개가 좋습니다.

| scorer                 | 설명                                     |
| ---------------------- | -------------------------------------- |
| Mahalanobis            | baseline과 연결 쉬움                        |
| kNN distance           | DCASE 상위권에서 자주 쓰이는 embedding 기반 방식과 맞음 |
| Gaussian density / GMM | source/target mixture로 확장 가능           |

처음에는 Mahalanobis와 kNN만 해도 충분합니다.

---

## Step 3. LCMED-ASD

LCMED는 다음 ablation으로 가면 됩니다.

| 실험            | 설명                                |
| ------------- | --------------------------------- |
| Fixed MED     | `[-1,0,1,300,500]`                |
| LCMED-2       | 300/500 branch만 learnable         |
| LCMED-4       | 100/300/500/1000 branch learnable |
| LCMED-channel | near/far/diff별 cutoff 따로 학습       |
| LCMED-shared  | near/far/diff가 cutoff 공유          |

가장 먼저 추천하는 것은 **LCMED-2 shared**입니다. 너무 많은 cutoff를 학습하면 target 10개 조건에서 과적합 위험이 큽니다.

---

# 8. SincNet 코드는 어디에 쓰면 좋은가

첨부하신 `compute_d_vector.py`는 SincNet pretrained model로 d-vector를 계산하는 script입니다. 이 코드는 직접 DCASE ASD에 바로 쓰기보다는, 다음 두 가지 용도로 보는 것이 좋습니다.

1. **learnable cutoff 설계 참고**
2. **embedding extraction pipeline 참고**

SincNet 자체는 speaker recognition용이라 기계음 ASD에 바로 적용하면 domain mismatch가 큽니다. 하지만 SincNet의 철학, 즉 “필터 전체를 학습하지 않고 cutoff frequency를 학습한다”는 점은 LCMED의 이론적 근거로 매우 좋습니다. SincNet 논문은 이 방식이 parameter 수를 줄이고, filter bank가 명확한 물리적 의미를 갖는다고 설명합니다. 

---

# 9. 논문/기술보고서 스토리라인

보고서 스토리는 이렇게 잡으면 좋습니다.

> DCASE 2026 Task 2는 near/far two-channel recording을 제공하므로, target machine component와 background noise 사이의 SNR 및 spectral difference를 활용할 수 있다. 기존 baseline은 log-mel reconstruction 및 Mahalanobis score에 기반하지만, 기계음의 envelope modulation과 long-term amplitude structure를 명시적으로 모델링하지 않는다. 본 연구는 BemaGANv2의 Multi-Envelope Discriminator를 vocoder discriminator에서 anomaly-aware feature extractor로 재해석한다. 또한 fixed cutoff의 한계를 보완하기 위해 Learnable-Cutoff MED를 제안하여 machine type 및 acoustic condition에 적응 가능한 envelope representation을 학습한다.

이렇게 쓰면 novelty는 다음 두 개가 됩니다.

1. **MED를 vocoder discriminator에서 ASD feature extractor로 전환**
2. **learnable cutoff envelope branch를 통해 noise-aware UASD에 적응**

---

# 10. 현실적인 구현 계획

파일 단위로는 다음처럼 가시면 됩니다.

```text
models/
  med_asd.py              # MEDFeatureExtractor
  lcmed_asd.py            # LearnableSoftEnvelope + LCMEDFeatureExtractor
scoring/
  mahalanobis.py          # source/target mean-cov scoring
  knn.py                  # optional
datasets/
  dcase2026_twoch.py      # baseline loader 기반 수정
train_med_asd.py          # feature extraction + scorer fitting
eval_med_asd.py           # test anomaly score csv 생성
```

기존 baseline 코드에서는 다음을 재사용하시면 됩니다.

| 파일                          | 재사용 목적                                  |
| --------------------------- | --------------------------------------- |
| `dcase2023t2_ae.py`         | train/test 흐름, source/target score 처리   |
| `mahala.py`                 | Mahalanobis 계산                          |
| `base_model.py`             | threshold fitting, result csv 저장 구조     |
| `datasets.py`, loader 관련 파일 | DCASE 파일명 파싱, domain/attribute handling |
| `dnn_models.py`             | LCMED 설계 참고용, 직접 사용 우선순위 낮음             |
| `compute_d_vector.py`       | embedding extraction script 구조 참고       |

---

# 11. 가장 먼저 해야 할 수정

가장 먼저 해야 할 것은 `Envelope` 교체입니다.

현재 `Envelope`는 다음 문제가 있습니다.

1. `.item()` 사용으로 cutoff gradient가 끊김
2. `torch.tensor(...)` 재생성으로 graph가 끊김
3. `len(signal)`이 batch dimension을 참조할 가능성이 큼
4. sample rate가 24000으로 고정되어 있음
5. DCASE 2026은 보통 16 kHz 기준이므로 `sr`를 외부 인자로 받아야 함

따라서 `Envelope`를 다음 두 개로 분리하는 것을 추천드립니다.

```python
FixedEnvelope      # MED-ASD용
LearnableEnvelope  # LCMED-ASD용
```

그리고 `DiscriminatorE`도 envelope module을 주입받도록 바꾸면 좋습니다.

```python
class DiscriminatorEForASD(nn.Module):
    def __init__(self, envelope_module):
        super().__init__()
        self.envelope = envelope_module
        ...
    
    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.envelope(x)
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        ...
        return x, fmap
```

이렇게 하면 fixed MED와 LCMED를 같은 구조로 비교할 수 있습니다.

---

# 결론

현재 첨부해주신 코드로 판단하면, **MED-ASD는 기존 `DiscriminatorE`와 baseline Mahalanobis 구조를 연결하면 바로 시작할 수 있습니다.** 다만 `Envelope` 구현은 DCASE waveform shape와 sample rate에 맞게 정리하는 것이 좋습니다.

반면 **LCMED-ASD는 현재 `envelope.py`로는 어렵고, differentiable soft low-pass envelope 또는 SincNet-style learnable filter로 새로 구현해야 합니다.** 가장 안전한 구현은 FFT 기반 soft low-pass mask입니다.

추천 순서는 다음입니다.

```text
1. DCASE baseline 실행
2. two-channel loader 확인
3. Fixed MED-ASD + Mahalanobis
4. Fixed MED-ASD + kNN
5. LCMED-ASD: 300/500 branch만 learnable
6. contrastive 또는 source-target alignment로 LCMED fine-tuning
7. score-level fusion: baseline AE + MED-ASD + LCMED-ASD
```

이렇게 가면 **기존 BemaGANv2 자산을 활용하면서도 DCASE 2026의 two-channel noise-aware 설정에 맞는 독자적인 방법론**으로 정리할 수 있습니다.

[1]: https://dcase.community/challenge2026/index?utm_source=chatgpt.com "DCASE2026 Challenge"
[2]: https://zenodo.org/records/19336329?utm_source=chatgpt.com "DCASE 2026 Challenge Task 2 Development Dataset"
[3]: https://github.com/nttcslab/dcase2023_task2_baseline_ae?utm_source=chatgpt.com "nttcslab/dcase2023_task2_baseline_ae"
